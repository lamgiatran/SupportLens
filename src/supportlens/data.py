"""Read raw MultiDoc2Dial without importing its executable dataset loader."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterator

from .text import Chunk


@dataclass(frozen=True)
class Document:
    domain: str
    doc_id: str
    title: str
    text: str


def documents(path: str | Path) -> Iterator[Document]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["doc_data"]
    for domain, docs in raw.items():
        for doc_id, doc in docs.items():
            yield Document(domain, doc_id, doc["title"], doc["doc_text"])


def examples(path: str | Path, *, dialogue_ids: set[str] | None = None) -> Iterator[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["dial_data"]
    for domain, dialogues in raw.items():
        for dialogue in dialogues:
            # Filter before constructing examples: ablations never evaluate held-out turns.
            if dialogue_ids is not None and dialogue["dial_id"] not in dialogue_ids:
                continue
            history = []
            turns = dialogue["turns"]
            for i, turn in enumerate(turns):
                if turn["role"] == "user" and i + 1 < len(turns):
                    target = turns[i + 1]
                    if target["role"] == "agent":
                        yield {
                            "query_id": f"{dialogue['dial_id']}:{turn['turn_id']}",
                            "dialogue_id": dialogue["dial_id"], "domain": domain,
                            "question": turn["utterance"], "history": list(history),
                            # Offline labels ONLY. Never send this field to retrieval.
                            "gold_references": target.get("references", []),
                        }
                history.append({"role": "assistant" if turn["role"] == "agent" else "user",
                                "content": turn["utterance"]})


class QrelsMapper:
    """Half-open raw character offsets; titles never count as span coverage.

    Strict: >= min_span_coverage of any one referenced span. Lenient: any
    positive overlap. Multiple references form a union of relevant chunks;
    this is retrieval relevance, not proof that all answer claims are covered.
    """

    def __init__(self, raw_docs: dict, chunks: list[Chunk], min_span_coverage: float = 0.5):
        if not 0 < min_span_coverage <= 1:
            raise ValueError("span coverage must be in (0,1]")
        self.docs, self.minimum = raw_docs, min_span_coverage
        self.by_doc: dict[tuple[str, str], list[Chunk]] = {}
        for chunk in chunks:
            self.by_doc.setdefault((chunk.domain, chunk.doc_id), []).append(chunk)
        self.cache = {}

    def _reference(self, domain: str, ref: dict) -> dict:
        doc_id, span_id = ref.get("doc_id"), str(ref.get("id_sp", ""))
        key = (domain, doc_id, span_id)
        if key in self.cache:
            return self.cache[key]
        result = {"doc_id": doc_id, "span_id": span_id, "strict": [], "lenient": [],
                  "coverage": [], "issue": None}
        doc = self.docs.get(domain, {}).get(doc_id)
        span = doc.get("spans", {}).get(span_id) if doc else None
        if span is None:
            result["issue"] = "unknown_span"
        else:
            start, end = span.get("start_sp"), span.get("end_sp")
            if (not isinstance(start, int) or not isinstance(end, int)
                    or not 0 <= start < end <= len(doc["doc_text"])):
                result["issue"] = "invalid_offsets"
            elif "text_sp" in span and doc["doc_text"][start:end] != span["text_sp"]:
                result["issue"] = "span_text_mismatch"
            else:
                result.update(start=start, end=end)
                for chunk in self.by_doc.get((domain, doc_id), []):
                    overlap = max(0, min(end, chunk.end) - max(start, chunk.start))
                    if overlap:
                        ratio = overlap / (end - start)
                        result["lenient"].append(chunk.chunk_id)
                        if ratio >= self.minimum:
                            result["strict"].append(chunk.chunk_id)
                        result["coverage"].append({"chunk_id": chunk.chunk_id,
                            "chunk_start": chunk.start, "chunk_end": chunk.end,
                            "span_fraction": ratio, "full_span": overlap == end - start})
                if not result["lenient"]:
                    result["issue"] = "no_chunk_overlap"
                elif not result["strict"]:
                    result["issue"] = "below_strict_coverage"
        self.cache[key] = result
        return result

    def map(self, domain: str, refs: list[dict]) -> dict:
        seen, references = set(), []
        for ref in refs:
            key = (ref.get("doc_id"), str(ref.get("id_sp", "")))
            if key not in seen:
                references.append(self._reference(domain, ref))
                seen.add(key)
        strict = sorted({c for r in references for c in r["strict"]})
        lenient = sorted({c for r in references for c in r["lenient"]})
        return {"strict": strict, "lenient": lenient, "references": references,
                "issues": [r for r in references if r["issue"] is not None]}


def map_references(domain: str, refs: list[dict], raw_docs: dict,
                   chunks: list[Chunk], *, relevance: str = "strict") -> tuple[set[str], list[dict]]:
    if relevance not in {"strict", "lenient"}:
        raise ValueError("relevance must be strict or lenient")
    result = QrelsMapper(raw_docs, chunks).map(domain, refs)
    missing = [{"reference": {"doc_id": r["doc_id"], "id_sp": r["span_id"]},
                "reason": r["issue"]} for r in result["issues"]
               if relevance == "strict" or not r["lenient"]]
    return set(result[relevance]), missing


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row,
                                    ensure_ascii=False) + "\n")
