"""Small, explicit Ollama raw-prompt probe. No generation latency is fabricated."""

from dataclasses import dataclass
import json
from time import perf_counter

import httpx

from .text import Tokenizer, prefix_within


@dataclass(frozen=True)
class Prompt:
    text: str
    tokens: int
    evidence_tokens: int


def make_prompt(question: str, evidence: list[dict], tokenizer: Tokenizer,
                num_ctx: int = 2048, num_predict: int = 128,
                evidence_budget: int = 600) -> Prompt:
    system = ("Answer only using the supplied evidence. Cite evidence IDs in brackets. "
              "If evidence is insufficient, say you cannot answer and request human review.")
    parts = []
    for item in evidence[:3]:
        candidate = "\n\n".join(parts + [f"[{item['evidence_id']}] {item['text']}"])
        if tokenizer.count(candidate, special=False) > evidence_budget:
            # Keep complete citation IDs; shorten body rather than cutting the whole prompt.
            header = f"[{item['evidence_id']}] "
            previous = "\n\n".join(parts)
            prefix = (previous + "\n\n" if previous else "") + header
            room = evidence_budget - tokenizer.count(prefix, special=False)
            if room > 0:
                body = prefix_within(item["text"], tokenizer, room, special=False)
                while body and tokenizer.count(prefix + body, special=False) > evidence_budget:
                    body = body[:-1]
                if body:
                    parts.append(header + body)
            break
        parts.append(f"[{item['evidence_id']}] {item['text']}")
    context = "\n\n".join(parts)
    # raw=True avoids hidden chat-template token accounting in this probe.
    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
              f"<|im_start|>user\nEvidence:\n{context}\n\nQuestion: {question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    tokens = tokenizer.count(prompt, special=False)
    if tokens + num_predict + 32 > num_ctx:
        raise ValueError("prompt + output reserve exceeds num_ctx; do not allow silent truncation")
    return Prompt(prompt, tokens, tokenizer.count(context, special=False))


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
    return {"status": "complete" if final.get("done") else "incomplete", "answer": text,
            "metric_scope": "generator_http_only_not_ui", "ttft_ms": first_ms,
            "total_ms": (perf_counter() - started) * 1000, "expected_prompt_tokens": prompt.tokens,
            "prompt_eval_count": actual, "prompt_token_audit": audit, "num_ctx": num_ctx,
            "num_predict": num_predict, "evidence_tokens": prompt.evidence_tokens,
            "raw_metrics": {k: final.get(k) for k in ("load_duration", "prompt_eval_duration", "eval_count", "eval_duration")}}
