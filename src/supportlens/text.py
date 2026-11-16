"""Token budgets use the encoder tokenizer, not word counts or LLM tokens."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol, Sequence


class Tokenizer(Protocol):
    def count(self, text: str, *, special: bool = True) -> int: ...
    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class JsonTokenizer:
    """A downloaded HF tokenizer.json; no model weights or remote code needed."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as HFTokenizer
        self.backend = HFTokenizer.from_file(str(path))
        self.backend.no_truncation()
        self.backend.no_padding()

    def count(self, text: str, *, special: bool = True) -> int:
        return len(self.backend.encode(text, add_special_tokens=special).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self.backend.encode(text, add_special_tokens=False).offsets


def prefix_within(text: str, tokenizer: Tokenizer, max_tokens: int, *, special=True) -> str:
    if tokenizer.count(text, special=special) <= max_tokens:
        return text
    offsets = tokenizer.offsets(text)
    allowance = max_tokens - (tokenizer.count("", special=True) if special else 0)
    if allowance <= 0:
        raise ValueError("token budget leaves no space for content")
    end = offsets[min(allowance, len(offsets)) - 1][1]
    candidate = text[:end]
    while candidate and tokenizer.count(candidate, special=special) > max_tokens:
        candidate = candidate[:-1]
    return candidate


@dataclass(frozen=True)
class Query:
    text: str
    original_tokens: int
    used_tokens: int
    truncated: bool
    current_question_truncated: bool


def build_query(question: str, history: Sequence[dict], tokenizer: Tokenizer,
                max_tokens: int = 256) -> Query:
    # history is chronological and contains past turns only. Current user first.
    # Formatting must not add lexical role labels or HF special-token literals.
    # The same packed text is passed to BM25 and every dense variant.
    parts = [question] + [t["content"] for t in reversed(history)]
    full = "\n".join(parts)
    packed = prefix_within(full, tokenizer, max_tokens)
    return Query(packed, tokenizer.count(full), tokenizer.count(packed),
                 packed != full, tokenizer.count(question) > max_tokens)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    domain: str
    doc_id: str
    title: str
    text: str
    start: int
    end: int
    tokens: int


def chunk_document(domain: str, doc_id: str, title: str, text: str,
                   tokenizer: Tokenizer, max_tokens: int = 220,
                   overlap: int = 32, encoder_limit: int = 256) -> list[Chunk]:
    if not 0 <= overlap < max_tokens <= encoder_limit:
        raise ValueError("require 0 <= overlap < chunk budget <= encoder limit")
    # Titles are part of the SAME text seen by BM25 and all dense variants.
    header = prefix_within(title, tokenizer, min(32, max_tokens // 3), special=False)
    offsets = tokenizer.offsets(text)
    chunks = []
    first = 0
    while first < len(offsets):
        last = min(len(offsets), first + max_tokens)
        while last > first:
            start, end = offsets[first][0], offsets[last - 1][1]
            value = f"{header}\n{text[start:end]}"
            size = tokenizer.count(value)
            if size <= max_tokens:
                break
            last -= 1
        if last == first:
            raise ValueError("one body token plus title exceeds chunk budget")
        uid = hashlib.sha256(f"{domain}\0{doc_id}\0{start}\0{end}".encode()).hexdigest()[:20]
        chunks.append(Chunk(uid, domain, doc_id, title, value, start, end, size))
        if last == len(offsets):
            break
        first = max(first + 1, last - overlap)
    return chunks
