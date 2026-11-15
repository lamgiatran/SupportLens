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


def examples(path: str | Path) -> Iterator[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["dial_data"]
    for domain, dialogues in raw.items():
        for dialogue in dialogues:
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


def map_references(domain: str, refs: list[dict], raw_docs: dict,
                   chunks: list[Chunk]) -> tuple[set[str], list[dict]]:
    positives, missing = set(), []
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.domain == domain:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
    for ref in refs:
        doc = raw_docs.get(domain, {}).get(ref["doc_id"])
        span = doc.get("spans", {}).get(str(ref["id_sp"])) if doc else None
        if span is None:
            missing.append({"reference": ref, "reason": "unknown_span"})
            continue
        start, end = span["start_sp"], span["end_sp"]
        if not 0 <= start < end <= len(doc["doc_text"]):
            missing.append({"reference": ref, "reason": "invalid_offsets"})
            continue
        hits = {c.chunk_id for c in by_doc.get(ref["doc_id"], [])
                if max(start, c.start) < min(end, c.end)}
        positives.update(hits)
        if not hits:
            missing.append({"reference": ref, "reason": "no_chunk_overlap"})
    return positives, missing


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row) if hasattr(row, "__dataclass_fields__") else row,
                                    ensure_ascii=False) + "\n")
