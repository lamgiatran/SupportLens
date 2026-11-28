"""Small, explicit Ollama raw-prompt probe. No generation latency is fabricated."""

from dataclasses import dataclass, field
import json
import re
from time import perf_counter

import httpx

from .text import Tokenizer, prefix_within


@dataclass(frozen=True)
class Prompt:
    text: str
    tokens: int
    evidence_tokens: int
    citation_map: dict[str, str] = field(default_factory=dict)
    citation_prefix: str = "S"
    visible_characters: dict[str, int] = field(default_factory=dict)


def resolve_citations(answer: str, prompt: Prompt) -> tuple[list[str], list[str]]:
    """Return real IDs plus unknown aliases; never silently accept invented IDs."""
    cited, invalid = [], []
    prefix = prompt.citation_prefix
    for group in re.findall(r"\[([^\[\]\r\n]+)\]", answer):
        # Bare numeric brackets belong to source text, not our citation grammar.
        if not re.search(rf"(?<!\w){re.escape(prefix)}\s*\d", group, re.I):
            continue
        for token in re.split(r"[,;]", group):
            token = token.strip()
            match = re.fullmatch(rf"(?:{re.escape(prefix)}\s*)?(\d+)", token, re.I)
            alias = prefix + str(int(match[1])) if match else token
            if alias in prompt.citation_map:
                value = prompt.citation_map[alias]
                if value not in cited:
                    cited.append(value)
            elif alias not in invalid:
                invalid.append(alias)
    return cited, invalid


def make_prompt(question: str, evidence: list[dict], tokenizer: Tokenizer,
                num_ctx: int = 2048, num_predict: int = 128,
                evidence_budget: int = 600, *, system: str | None = None,
                suffix: str | None = None, conversation: str = "") -> Prompt:
    """`system`/`suffix` are templates with `{p}` for the citation prefix; None keeps the v1 prompt."""
    # No namespace is inherently impossible in arbitrary source text. Reserve a
    # short one absent from the supplied text; never edit raw document brackets.
    original = question + "\n" + conversation + "\n" + "\n".join(item["text"] for item in evidence[:3])
    prefix = "S"
    while re.search(rf"\[[^\[\]\r\n]*\b{prefix}\s*\d", original, re.I):
        prefix += "S"
    default_system = (f"Answer only using the supplied evidence. Cite sources as [{prefix}1] or [{prefix}2]. "
              f"For multiple sources use [{prefix}1, {prefix}2]. "
              "Use only source aliases actually present below. Numeric brackets in source text are not citations. "
              "If evidence is insufficient, say you cannot answer and request human review.")
    system = default_system if system is None else system.format(p=prefix)
    tail = "" if suffix is None else "\n" + suffix.format(p=prefix)
    parts, citation_map, visible = [], {}, {}
    for number, item in enumerate(evidence[:3], 1):
        alias = prefix + str(number)
        evidence_id = item["evidence_id"]
        if not evidence_id or evidence_id in citation_map.values():
            raise ValueError("evidence IDs must be non-empty and unique")
        header = f"[{alias}] "
        candidate = "\n\n".join(parts + [header + item["text"]])
        if tokenizer.count(candidate, special=False) > evidence_budget:
            # Keep the complete source alias; trim only the evidence body.
            previous = "\n\n".join(parts)
            packed_prefix = (previous + "\n\n" if previous else "") + header
            room = evidence_budget - tokenizer.count(packed_prefix, special=False)
            if room > 0:
                body = prefix_within(item["text"], tokenizer, room, special=False)
                while body and tokenizer.count(packed_prefix + body, special=False) > evidence_budget:
                    body = body[:-1]
                if body:
                    parts.append(header + body)
                    citation_map[alias] = evidence_id
                    visible[evidence_id] = len(body)
            break
        parts.append(header + item["text"])
        citation_map[alias] = evidence_id
        visible[evidence_id] = len(item["text"])
    context = "\n\n".join(parts)
    # raw=True avoids hidden chat-template token accounting in this probe.
    background = f"\n\nPrior conversation (for meaning only):\n{conversation}" if conversation else ""
    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
              f"<|im_start|>user\nEvidence:\n{context}{background}\n\nQuestion: {question}{tail}<|im_end|>\n"
              "<|im_start|>assistant\n")
    tokens = tokenizer.count(prompt, special=False)
    if tokens + num_predict + 32 > num_ctx:
        raise ValueError("prompt + output reserve exceeds num_ctx; do not allow silent truncation")
    return Prompt(prompt, tokens, tokenizer.count(context, special=False), citation_map, prefix, visible)


def stream_probe(base_url: str, model: str, prompt: Prompt, *, num_ctx: int,
                 num_predict: int = 128, timeout: float = 45) -> dict:
    started = perf_counter()
    first_ms = None
    text, final = "", {}
    payload = {"model": model, "prompt": prompt.text, "raw": True, "stream": True,
               "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0}}
    with httpx.stream("POST", base_url.rstrip("/") + "/api/generate", json=payload, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(event["error"])
            token = event.get("response", "")
            if token and first_ms is None:
                first_ms = (perf_counter() - started) * 1000
            text += token
            if event.get("done"):
                final = event
    actual = final.get("prompt_eval_count")
    # GGUF/HF BOS handling can differ slightly. This is diagnostic, not a proof.
    audit = "unavailable" if actual is None else (
        "count_matches_within_2" if abs(actual - prompt.tokens) <= 2 else "token_count_mismatch")
    citations, invalid = resolve_citations(text, prompt)
    return {"status": "complete" if final.get("done") else "incomplete", "answer": text,
            "citations": citations, "invalid_citations": invalid,
            "citation_map": prompt.citation_map,
            "citation_status": "invalid" if invalid else ("present" if citations else "missing"),
            "metric_scope": "generator_http_only_not_ui", "ttft_ms": first_ms,
            "total_ms": (perf_counter() - started) * 1000, "expected_prompt_tokens": prompt.tokens,
            "prompt_eval_count": actual, "prompt_token_audit": audit, "num_ctx": num_ctx,
            "num_predict": num_predict, "evidence_tokens": prompt.evidence_tokens,
            "raw_metrics": {k: final.get(k) for k in ("load_duration", "prompt_eval_duration", "eval_count", "eval_duration")}}
