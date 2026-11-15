"""R1 document retrieval for the day-two vertical slice; not a dense benchmark."""

import hashlib
import json
import re
from pathlib import Path

from .contracts import Evidence


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
