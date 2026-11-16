"""R1 document retrieval for the day-two vertical slice; not a dense benchmark."""

import hashlib
import json
import re
from pathlib import Path
from dataclasses import asdict

from .contracts import Evidence
from .text import Chunk, JsonTokenizer, build_query


def words(value: str) -> list[str]:
    return re.findall(r"\w+", value.lower())


class BM25Documents:
    def __init__(self, rows: list[dict]):
        from rank_bm25 import BM25Okapi
        if not rows:
            raise ValueError("empty corpus")
        self.rows = rows
        self.index = BM25Okapi([words(r["title"] + "\n" + r["text"]) for r in rows])
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.version = "R1-document-" + digest[:16]

    @classmethod
    def from_jsonl(cls, path: str | Path):
        return cls([json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line])

    def search(self, question: str, k: int = 5) -> list[Evidence]:
        scores = self.index.get_scores(words(question))
        # Do not equate score zero with no answer: policy thresholds need dev data.
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [Evidence(evidence_id=f"doc:{i}", doc_id=self.rows[i]["doc_id"],
                         domain=self.rows[i]["domain"], title=self.rows[i]["title"],
                         text=self.rows[i]["text"], score=float(scores[i]), unit="document")
                for i in order]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, key in enumerate(ranking, 1):
            if key not in seen:
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                seen.add(key)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def deduplicate_chunks(ranked: list[tuple[Chunk, float]], k: int,
                       overlap_threshold: float = 0.5) -> list[tuple[Chunk, float]]:
    if k < 1 or not 0 < overlap_threshold <= 1:
        raise ValueError("positive k and overlap threshold in (0,1] required")
    selected, seen = [], set()
    for chunk, score in ranked:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        duplicate = False
        for other, _ in selected:
            if (chunk.domain, chunk.doc_id) == (other.domain, other.doc_id):
                overlap = max(0, min(chunk.end, other.end) - max(chunk.start, other.start))
                if overlap / min(chunk.end - chunk.start, other.end - other.start) >= overlap_threshold:
                    duplicate = True
                    break
        if not duplicate:
            selected.append((chunk, score))
            if len(selected) == k:
                break
    return selected


class BM25Chunks:
    """The API and offline R1 eval use the same text, query and dedup policy."""
    def __init__(self, chunks: list[Chunk], tokenizer: JsonTokenizer, *,
                 query_max_tokens: int = 256, dedup_overlap: float = 0.5):
        from rank_bm25 import BM25Okapi
        if (not chunks or len({c.chunk_id for c in chunks}) != len(chunks)
                or any(c.start >= c.end for c in chunks) or not 0 < dedup_overlap <= 1):
            raise ValueError("unique non-empty chunks and valid overlap policy required")
        self.chunks, self.tokenizer = chunks, tokenizer
        self.query_max_tokens, self.dedup_overlap = query_max_tokens, dedup_overlap
        # chunk.text already contains the title; never add it a second time.
        self.index = BM25Okapi([words(c.text) for c in chunks])
        policy = {"query": "strip_current_reverse_history_newline_v1", "limit": query_max_tokens,
                  "tokenizer_sha256": tokenizer.fingerprint, "dedup_overlap": dedup_overlap}
        digest = hashlib.sha256(json.dumps({"chunks": [asdict(c) for c in chunks], "policy": policy},
                                          sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.version = "R1-chunk-" + digest[:16]

    def prepare_query(self, question: str, history: list[dict]):
        return build_query(question, history, self.tokenizer, self.query_max_tokens)

    def search(self, query_text: str, k: int = 5) -> list[Evidence]:
        scores = self.index.get_scores(words(query_text))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        ranked = [(self.chunks[i], float(scores[i])) for i in order]
        selected = deduplicate_chunks(ranked, k, self.dedup_overlap)
        return [Evidence(evidence_id=c.chunk_id, doc_id=c.doc_id, domain=c.domain,
                         title=c.title, text=c.text, score=score, unit="chunk") for c, score in selected]

    @classmethod
    def from_bundle(cls, directory: str | Path, tokenizer_path: str | Path = "data/tokenizers/minilm/tokenizer.json"):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        tokenizer = JsonTokenizer(tokenizer_path)
        if tokenizer.fingerprint != manifest["tokenizer_sha256"]:
            raise ValueError("bundle tokenizer does not match the serving tokenizer")
        path = directory / "chunks.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files_sha256"]["chunks.jsonl"]:
            raise ValueError("chunk bytes do not match the bundle manifest")
        chunks = [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
        obj = cls(chunks, tokenizer, query_max_tokens=manifest["config"]["query_max_tokens"],
                  dedup_overlap=manifest["config"]["dedup_overlap"])
        if obj.version != manifest["artifact_version"]:
            raise ValueError("serving query/retrieval policy does not match the bundle")
        return obj
