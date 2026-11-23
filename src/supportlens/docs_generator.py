"""Docs-QA answer generation with source citations, an answer cache and a hard deadline.

An answer is published only when it cites at least one supplied passage and every
citation resolves to a passage that was visible in the prompt. Anything else
(no citation, invented citation, refusal, timeout, server error) falls back to the
retrieved passages or a human handoff; the generator never fabricates sources.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import logging
from pathlib import Path
import re
from threading import Event, Lock
from time import perf_counter
from typing import Callable, Iterator, Literal

import httpx

from .generation import Prompt, make_prompt, resolve_citations
from .text import Tokenizer

LOGGER = logging.getLogger("uvicorn.error.supportlens.docs_generator")
# Prompt policies are selected on dev. v1 is the original instruction; v2 asks for short,
# sentence-level citations with one example, repeated right after the question, because
# qwen2.5:1.5b answered from the evidence but rarely emitted labels under v1 (dev run 24/11).
PROMPTS = {
    "v1": (None, None),
    "v2": ("You answer questions using only the numbered sources below.\n"
           "Rules:\n"
           "1. Answer in at most 3 short sentences.\n"
           "2. End every sentence with the label of the source it came from, like [{p}1] or [{p}1, {p}2].\n"
           "3. Use only labels that appear below. Numbers inside the source text are not labels.\n"
           "4. If the sources do not contain the answer, reply exactly: "
           "I cannot answer from these sources; please request human review.\n"
           "Example answer: You can renew online or at an office [{p}1]. The fee is $25 [{p}2].",
           "Answer in at most 3 sentences and end each sentence with its source label, e.g. [{p}1]."),
}
WARMUP_QUESTION = "Which source explains this demo?"
WARMUP_EVIDENCE = [{"evidence_id": "warmup-1", "text": "This synthetic passage only warms up the generator."}]
REFUSAL = re.compile(r"cannot answer|can't answer|unable to answer|not enough information|"
                     r"insufficient (?:evidence|information)|human review", re.I)

Status = Literal["answer", "refused", "no_citation", "invalid_citation", "empty",
                 "timeout", "cancelled", "error"]


@dataclass(frozen=True)
class GeneratorConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:1.5b"
    num_ctx: int = 2048
    num_predict: int = 128
    evidence_budget_tokens: int = 600
    evidence_top_k: int = 3
    temperature: float = 0.0
    deadline_seconds: float = 30.0
    connect_timeout_seconds: float = 3.0
    cache_size: int = 256
    prompt_policy: str = "v2"

    def __post_init__(self):
        if self.prompt_policy not in PROMPTS:
            raise ValueError(f"prompt_policy must be one of {sorted(PROMPTS)}")
        if self.evidence_top_k < 1 or self.evidence_top_k > 3:
            raise ValueError("make_prompt packs at most 3 passages; evidence_top_k must be 1..3")
        if self.deadline_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("deadline and connect timeout must be positive")
        if self.num_predict < 1 or self.num_ctx <= self.num_predict or self.cache_size < 0:
            raise ValueError("invalid generation budget")

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "GeneratorConfig":
        """Read the `generator` block of configs/default.json; unknown keys are ignored."""
        block = json.loads(Path(path).read_text(encoding="utf-8"))["generator"]
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in block.items() if k in known}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    @property
    def version(self) -> str:
        identity = {k: v for k, v in asdict(self).items()
                    if k not in {"base_url", "deadline_seconds", "connect_timeout_seconds", "cache_size"}}
        identity["prompt_template"] = "chatml_raw_cited_top3"
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return "G-" + digest[:16]


@dataclass
class GenerationResult:
    status: Status
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    cache_status: Literal["miss", "hit"] = "miss"
    ttft_ms: float | None = None
    generate_ms: float = 0.0
    error_type: str | None = None

    def decision(self) -> tuple[Literal["answer", "retrieved", "handoff"], str]:
        """Map a generation outcome to the API decision and a machine-readable reason."""
        if self.status == "answer":
            return "answer", "generated_with_verified_citations"
        if self.status == "refused":
            return "handoff", "generator_insufficient_evidence"
        return "retrieved", f"generator_{self.status}_fallback_to_evidence"


def classify(text: str, prompt: Prompt) -> tuple[Status, list[str], list[str]]:
    citations, invalid = resolve_citations(text, prompt)
    if not text.strip():
        return "empty", [], invalid
    if invalid:
        return "invalid_citation", citations, invalid
    if not citations:
        return ("refused" if REFUSAL.search(text) else "no_citation"), [], []
    return "answer", citations, []


class OllamaGenerator:
    """Streams from a local Ollama server. Startup checks the model once; requests never pull."""

    def __init__(self, config: GeneratorConfig, tokenizer: Tokenizer, *,
                 transport: httpx.BaseTransport | None = None,
                 clock: Callable[[], float] = perf_counter):
        self.config, self.tokenizer, self.clock = config, tokenizer, clock
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"), transport=transport,
            timeout=httpx.Timeout(config.deadline_seconds, connect=config.connect_timeout_seconds))
        self.version = config.version
        self.state, self.error_type = "pending", None
        self.warmup_ms: float | None = None
        self._cache: OrderedDict[str, GenerationResult] = OrderedDict()
        self._cache_lock, self._start_lock = Lock(), Lock()

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def snapshot(self) -> dict:
        return {"state": self.state, "model": self.config.model, "version": self.version,
                "error_type": self.error_type, "warmup_ms": self.warmup_ms,
                "deadline_seconds": self.config.deadline_seconds, "cache_entries": len(self._cache)}

    def start(self) -> None:
        """Check the pinned model exists, then load it with one short generation."""
        with self._start_lock:
            if self.state != "pending":
                return
            self.state, started = "loading", self.clock()
            try:
                tags = self.client.get("/api/tags", timeout=self.config.connect_timeout_seconds)
                tags.raise_for_status()
                names = {m.get("name") for m in tags.json().get("models", [])}
                if self.config.model not in names:
                    raise LookupError(f"model {self.config.model!r} is not pulled; run: ollama pull {self.config.model}")
                result = self.generate(WARMUP_QUESTION, WARMUP_EVIDENCE, use_cache=False)
                if result.status in {"timeout", "error"}:
                    raise RuntimeError(f"warm-up generation ended with {result.status}: {result.error_type}")
                self.warmup_ms = (self.clock() - started) * 1000
                self.state = "ready"
            except Exception as error:
                self.error_type, self.state = type(error).__name__, "failed"
                LOGGER.exception("Docs generator startup failed")
            finally:
                LOGGER.info("Docs generator startup: %s", self.snapshot())

    def build_prompt(self, question: str, evidence: list) -> Prompt:
        items = [e if isinstance(e, dict) else {"evidence_id": e.evidence_id, "text": e.text}
                 for e in evidence[:self.config.evidence_top_k]]
        system, suffix = PROMPTS[self.config.prompt_policy]
        return make_prompt(question, items, self.tokenizer, self.config.num_ctx,
                           self.config.num_predict, self.config.evidence_budget_tokens,
                           system=system, suffix=suffix)

    def _cache_key(self, prompt: Prompt) -> str:
        return hashlib.sha256((self.version + "\n" + prompt.text).encode()).hexdigest()

    def stream(self, question: str, evidence: list, *, cancel: Event | None = None,
               use_cache: bool = True) -> Iterator[dict]:
        """Yield {"event": "delta", "text"} items, then exactly one {"event": "result", "result"}."""
        started = self.clock()
        try:
            prompt = self.build_prompt(question, evidence)
        except ValueError as error:
            yield {"event": "result", "result": GenerationResult("error", error_type=type(error).__name__)}
            return
        key = self._cache_key(prompt)
        if use_cache and self.config.cache_size:
            with self._cache_lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
            if cached is not None:
                yield {"event": "delta", "text": cached.answer}
                hit = GenerationResult(**{**asdict(cached), "cache_status": "hit", "ttft_ms": 0.0,
                                          "generate_ms": (self.clock() - started) * 1000})
                yield {"event": "result", "result": hit}
                return
        payload = {"model": self.config.model, "prompt": prompt.text, "raw": True, "stream": True,
                   "options": {"num_ctx": self.config.num_ctx, "num_predict": self.config.num_predict,
                               "temperature": self.config.temperature}}
        text, first, status, error_type = "", None, None, None
        try:
            with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if cancel is not None and cancel.is_set():
                        status = "cancelled"
                        break
                    if self.clock() - started > self.config.deadline_seconds:
                        status = "timeout"
                        break
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    token = event.get("response", "")
                    if token:
                        if first is None:
                            first = (self.clock() - started) * 1000
                        text += token
                        yield {"event": "delta", "text": token}
                    if event.get("done"):
                        break
                else:
                    status = "error"
                    error_type = "StreamEndedWithoutDone"
        except httpx.TimeoutException as error:
            status, error_type = "timeout", type(error).__name__
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            status, error_type = "error", type(error).__name__
        elapsed = (self.clock() - started) * 1000
        if status is None:
            status, citations, invalid = classify(text, prompt)
        else:
            citations, invalid = [], []
        result = GenerationResult(status, text if status not in {"timeout", "cancelled", "error"} else "",
                                  citations, invalid, "miss", first, elapsed, error_type)
        if use_cache and self.config.cache_size and status in {"answer", "refused"}:
            with self._cache_lock:
                self._cache[key] = result
                while len(self._cache) > self.config.cache_size:
                    self._cache.popitem(last=False)
        yield {"event": "result", "result": result}

    def generate(self, question: str, evidence: list, *, use_cache: bool = True) -> GenerationResult:
        result = None
        for event in self.stream(question, evidence, use_cache=use_cache):
            if event["event"] == "result":
                result = event["result"]
        return result


def generator_from_env(environ) -> OllamaGenerator | None:
    """Explicit opt-in. A broken configuration fails startup visibly instead of silently disabling."""
    mode = environ.get("SUPPORTLENS_GENERATOR", "disabled")
    if mode == "disabled":
        return None
    if mode != "ollama":
        raise ValueError("SUPPORTLENS_GENERATOR must be disabled or ollama")
    from .text import JsonTokenizer
    deadline = environ.get("SUPPORTLENS_GENERATOR_DEADLINE")
    config = GeneratorConfig.load(environ.get("SUPPORTLENS_GENERATOR_CONFIG", "configs/default.json"),
                                  base_url=environ.get("SUPPORTLENS_OLLAMA_URL"),
                                  deadline_seconds=float(deadline) if deadline else None)
    tokenizer = JsonTokenizer(environ.get("SUPPORTLENS_GENERATOR_TOKENIZER", "data/tokenizers/qwen/tokenizer.json"))
    return OllamaGenerator(config, tokenizer)
