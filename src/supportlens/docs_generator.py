"""Docs-QA answer generation with source citations, an answer cache and a hard deadline.

An answer is published only when it cites at least one supplied passage and every
citation resolves to a passage that was visible in the prompt. Anything else
(no citation, invented citation, refusal, timeout, server error) falls back to the
retrieved passages or a human handoff; the generator never fabricates sources.
"""

import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
import json
import logging
from pathlib import Path
from queue import Empty, SimpleQueue
import re
from threading import Event, Lock
from time import perf_counter
from typing import Callable, Iterator, Literal

import httpx
from anyio.from_thread import start_blocking_portal

from .generation import Prompt, make_prompt, resolve_citations
from .generation_journal import GenerationJournal, identity_hash
from .grounded_answer import check_cited_values, copied_example, recent_conversation
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
    "v3": ("Answer the user's current question directly from the numbered evidence. Use the prior conversation "
           "only to identify what the user is referring to. It is not a source. Give one short answer, "
           "including any condition needed to make it true. Omit unrelated instructions and additional "
           "fees or telephone numbers. Cite each factual statement with its own source label before the "
           "period, for example: You can apply online [{p}2]. If the evidence does not answer this specific "
           "request, say exactly: I cannot answer from these sources; please request human review.",
           "Give the answer and cite the passage that actually states it. If the answer is missing, request human review."),
}
WARMUP_QUESTION = "Which source explains this demo?"
PROMPTS["v2_checked"] = PROMPTS["v2"]
WARMUP_EVIDENCE = [{"evidence_id": "warmup-1", "text": "This synthetic passage only warms up the generator."}]
REFUSAL = re.compile(r"cannot answer|can't answer|unable to answer|not enough information|"
                     r"insufficient (?:evidence|information)|human review", re.I)

Status = Literal["answer", "refused", "no_citation", "invalid_citation", "empty",
                 "timeout", "cancelled", "error", "unverified_claim"]


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
    warmup_deadline_seconds: float = 120.0
    connect_timeout_seconds: float = 3.0
    cache_size: int = 256
    prompt_policy: str = "v2"

    def __post_init__(self):
        if self.prompt_policy not in PROMPTS:
            raise ValueError(f"prompt_policy must be one of {sorted(PROMPTS)}")
        if self.evidence_top_k < 1 or self.evidence_top_k > 3:
            raise ValueError("make_prompt packs at most 3 passages; evidence_top_k must be 1..3")
        if min(self.deadline_seconds, self.warmup_deadline_seconds, self.connect_timeout_seconds) <= 0:
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
                    if k not in {"base_url", "deadline_seconds", "warmup_deadline_seconds", "connect_timeout_seconds", "cache_size"}}
        identity["prompt_template"] = "chatml_raw_cited_top3"
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return "G-" + digest[:16]


@dataclass
class GenerationResult:
    status: Status
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    cache_status: Literal["miss", "hit", "replay"] = "miss"
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
                 transport: httpx.MockTransport | None = None, journal_dir: str | Path | None = None,
                 include_prompt: bool = False, clock: Callable[[], float] = perf_counter):
        self.config, self.tokenizer, self.clock = config, tokenizer, clock
        self.transport = transport
        self.journal = GenerationJournal(journal_dir) if journal_dir is not None else None
        self.include_prompt = include_prompt
        self.model_digest = None
        # Reuse TLS trust roots across per-request event loops, especially on Windows.
        self._verify = httpx.create_ssl_context() if transport is None else True
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"), transport=transport, verify=self._verify,
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
                "model_digest": self.model_digest,
                "error_type": self.error_type, "warmup_ms": self.warmup_ms,
                "deadline_seconds": self.config.deadline_seconds, "warmup_deadline_seconds": self.config.warmup_deadline_seconds,
                "cache_entries": len(self._cache)}

    def start(self) -> None:
        """Check the pinned model exists, then load it with one short generation."""
        with self._start_lock:
            if self.state != "pending":
                return
            self.state, started = "loading", self.clock()
            try:
                tags = self.client.get("/api/tags", timeout=self.config.connect_timeout_seconds)
                tags.raise_for_status()
                models = {m.get("name"): m.get("digest") for m in tags.json().get("models", [])}
                if self.config.model not in models:
                    raise LookupError(f"model {self.config.model!r} is not pulled; run: ollama pull {self.config.model}")
                self.model_digest = models[self.config.model]
                result = self.generate(WARMUP_QUESTION, WARMUP_EVIDENCE, use_cache=False, warmup=True)
                if result.status in {"timeout", "error"}:
                    raise RuntimeError(f"warm-up generation ended with {result.status}: {result.error_type}")
                self.warmup_ms = (self.clock() - started) * 1000
                self.state = "ready"
            except Exception as error:
                self.error_type, self.state = type(error).__name__, "failed"
                LOGGER.exception("Docs generator startup failed")
            finally:
                LOGGER.info("Docs generator startup: %s", self.snapshot())

    def build_prompt(self, question: str, evidence: list, *, history=()) -> Prompt:
        items = [e if isinstance(e, dict) else {"evidence_id": e.evidence_id, "text": e.text}
                 for e in evidence[:self.config.evidence_top_k]]
        system, suffix = PROMPTS[self.config.prompt_policy]
        return make_prompt(question, items, self.tokenizer, self.config.num_ctx,
                           self.config.num_predict, self.config.evidence_budget_tokens,
                           system=system, suffix=suffix,
                           conversation=recent_conversation(history, self.tokenizer)
                           if self.config.prompt_policy == "v3" else "")

    async def _read_ollama(self, payload, timeout, messages):
        # The scope includes connecting, waiting for headers and waiting between tokens.
        try:
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(base_url=self.config.base_url, transport=self.transport, verify=self._verify,
                                             timeout=httpx.Timeout(timeout, connect=min(timeout, self.config.connect_timeout_seconds))) as client:
                    async with client.stream("POST", "/api/generate", json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            event = json.loads(line)
                            if "error" in event:
                                raise RuntimeError(event["error"])
                            messages.put(event)
                            if event.get("done"):
                                return
            raise RuntimeError("StreamEndedWithoutDone")
        except (TimeoutError, httpx.TimeoutException) as error:
            messages.put(("timeout", type(error).__name__))
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            messages.put(("error", str(error) if str(error) == "StreamEndedWithoutDone" else type(error).__name__))
        finally:
            messages.put(None)

    def _live_events(self, payload, deadline, cancel):
        """Bridge the cancellable HTTP stream to the sync API and CLI."""
        messages = SimpleQueue()
        with start_blocking_portal() as portal:
            reader = portal.start_task_soon(self._read_ollama, payload, max(0, deadline - self.clock()), messages)
            try:
                while True:
                    if cancel is not None and cancel.is_set():
                        yield ("cancelled", None)
                        break
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        yield ("timeout", "TimeoutError")
                        break
                    try:
                        event = messages.get(timeout=min(remaining, 0.05))
                    except Empty:
                        continue
                    if event is None:
                        break
                    yield event
                    if isinstance(event, tuple) or event.get("done"):
                        break
            finally:
                reader.cancel()

    def _remember(self, key, result):
        with self._cache_lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)

    def _lookup(self, key):
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return replace(cached, cache_status="hit")
        if self.journal is not None and self.model_digest:
            saved = self.journal.replay(key)
            if saved is not None:
                result = GenerationResult(**saved)
                self._remember(key, result)
                return replace(result, cache_status="replay")
        return None

    def stream(self, question: str, evidence: list, *, cancel: Event | None = None,
               use_cache: bool = True, history=(), retrieval_version=None, warmup=False) -> Iterator[dict]:
        """Yield provisional deltas followed by one terminal result."""
        started = self.clock()
        items = [e if isinstance(e, dict) else {"evidence_id": e.evidence_id, "text": e.text}
                 for e in evidence[:self.config.evidence_top_k]]
        identity = {"schema": 1, "generator": self.version, "model": self.config.model,
                    "model_digest": self.model_digest, "tokenizer": getattr(self.tokenizer, "fingerprint", None),
                    "retrieval_version": retrieval_version, "question_sha256": identity_hash(question),
                    "history_sha256": identity_hash(list(history)),
                    "evidence": [[e["evidence_id"], identity_hash(e["text"])] for e in items]}
        contextual = self.config.prompt_policy == "v3"
        guarded = contextual or self.config.prompt_policy == "v2_checked"
        prompt, final = None, {}
        text, first, status, error_type = "", None, None, None
        journal = self.journal if not warmup else None
        try:
            prompt = self.build_prompt(question, items, history=history) if contextual else self.build_prompt(question, items)
        except ValueError as error:
            status, error_type = "error", type(error).__name__
        identity["prompt_sha256"] = identity_hash(prompt.text) if prompt else None
        key = identity_hash(identity)
        caching = use_cache and self.config.cache_size > 0 and prompt is not None
        cached = self._lookup(key) if caching else None
        if cached is not None:
            if not guarded or cached.status == "answer":
                yield {"event": "delta", "text": cached.answer}
            yield {"event": "result", "result": replace(cached, ttft_ms=None, generate_ms=(self.clock() - started) * 1000)}
            return

        try:
            if prompt is not None:
                payload = {"model": self.config.model, "prompt": prompt.text, "raw": True, "stream": True,
                           "options": {"num_ctx": self.config.num_ctx, "num_predict": self.config.num_predict,
                                       "temperature": self.config.temperature}}
                budget = self.config.warmup_deadline_seconds if warmup else self.config.deadline_seconds
                events = self._live_events(payload, started + budget, cancel)
                try:
                    for event in events:
                        if isinstance(event, tuple):
                            status, error_type = event
                            break
                        token = event.get("response", "")
                        if token:
                            if first is None and not guarded:
                                first = (self.clock() - started) * 1000
                            text += token
                            if not guarded:
                                yield {"event": "delta", "text": token}
                        if event.get("done"):
                            final = event
                finally:
                    events.close()
                if status is None and not final:
                    status, error_type = "error", "StreamEndedWithoutDone"
        except GeneratorExit:
            status = "cancelled"
            raise
        finally:
            citations, invalid = [], []
            if status is None:
                status, citations, invalid = classify(text, prompt)
                if guarded and status == "answer":
                    if contextual and REFUSAL.search(text):
                        status, citations = "refused", []
                    elif ((contextual and not check_cited_values(text, prompt, items))
                          or (not contextual and copied_example(text, prompt, items))):
                        status, citations = "unverified_claim", []
                    else:
                        first = (self.clock() - started) * 1000
            result = GenerationResult(status, "" if status in {"timeout", "cancelled", "error", "unverified_claim"} else text,
                                      citations, invalid, "miss", first, (self.clock() - started) * 1000, error_type)
            if journal is not None:
                packed = asdict(prompt) if prompt else {}
                if not self.include_prompt:
                    packed.pop("text", None)
                row = {"key": key, "identity": identity, "options": asdict(self.config), "prompt": packed,
                       "raw_text": text, "ollama": {k: final.get(k) for k in ("prompt_eval_count", "eval_count",
                                    "load_duration", "prompt_eval_duration", "eval_duration", "done_reason")},
                       "result": asdict(result)}
                if self.include_prompt:
                    row["input"] = {"question": question, "history": list(history)}
                journal.save(key, row)
        if caching and status in {"answer", "refused"}:
            self._remember(key, result)
        if guarded and result.status == "answer":
            yield {"event": "delta", "text": result.answer}
        yield {"event": "result", "result": result}

    def generate(self, question: str, evidence: list, **kwargs) -> GenerationResult:
        for event in self.stream(question, evidence, **kwargs):
            if event["event"] == "result":
                return event["result"]
        raise RuntimeError("missing terminal generation result")


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
                                  prompt_policy=environ.get("SUPPORTLENS_GENERATOR_POLICY"),
                                  base_url=environ.get("SUPPORTLENS_OLLAMA_URL"),
                                  deadline_seconds=float(deadline) if deadline else None)
    tokenizer = JsonTokenizer(environ.get("SUPPORTLENS_GENERATOR_TOKENIZER", "data/tokenizers/qwen/tokenizer.json"))
    return OllamaGenerator(config, tokenizer, journal_dir=environ.get("SUPPORTLENS_GENERATION_DIR", "runs/generation_cache"),
                           include_prompt=environ.get("SUPPORTLENS_LOG_QUESTIONS", "0") == "1")
