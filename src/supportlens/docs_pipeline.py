"""Create corpus/chunks/qrels before model selection. No held-out retrieval run."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

from .data import QrelsMapper, examples
from .retrieval import BM25Chunks
from .text import JsonTokenizer, Query, build_query, chunk_document


DEFAULTS = {"chunk_max_tokens": 220, "chunk_overlap": 32, "encoder_limit": 256,
            "query_max_tokens": 256, "strict_min_span_coverage": 0.5,
            "dedup_overlap": 0.5, "validation_dev_fraction": 0.5, "seed": 42,
            "history_turns": None}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dialogue_split(train: dict, validation: dict, *, seed: int = 42, dev_fraction: float = 0.5) -> dict:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev fraction must be in (0,1)")
    train_ids = [d["dial_id"] for group in train.values() for d in group]
    val_ids = [d["dial_id"] for group in validation.values() for d in group]
    if (len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids))
            or set(train_ids) & set(val_ids)):
        raise ValueError("duplicate/overlapping dialogue IDs; resolve before splitting")
    roles = {"train": sorted(train_ids), "dev": [], "heldout": []}
    for domain, group in sorted(validation.items()):
        ids = sorted(d["dial_id"] for d in group)
        if len(ids) < 2:
            raise ValueError("each validation domain needs at least two dialogues")
        random.Random(f"{seed}:{domain}").shuffle(ids)
        cut = max(1, min(len(ids) - 1, int(len(ids) * dev_fraction)))
        roles["dev"].extend(ids[:cut])
        roles["heldout"].extend(ids[cut:])
    return {"seed": seed, "dev_fraction": dev_fraction,
            "scheme": "public train -> train; public validation -> domain-stratified dialogue dev/heldout",
            "dialogues": {k: sorted(v) for k, v in roles.items()}}


def _line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_docs_bundle(raw_dir: str | Path, tokenizer_path: str | Path, destination: str | Path,
                      *, config: dict | None = None) -> dict:
    started = perf_counter()
    if set(config or {}) - set(DEFAULTS):
        raise ValueError("unknown Docs-QA configuration field")
    cfg = DEFAULTS | (config or {})
    if not 1 <= cfg["query_max_tokens"] <= cfg["encoder_limit"]:
        raise ValueError("query budget must not exceed the encoder limit")
    if cfg["history_turns"] is not None and (type(cfg["history_turns"]) is not int or cfg["history_turns"] < 0):
        raise ValueError("history_turns must be a non-negative integer or None")
    raw_dir, destination = Path(raw_dir), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sources = {role: raw_dir / name for role, name in {
        "documents": "multidoc2dial_doc.json", "train": "multidoc2dial_dial_train.json",
        "validation": "multidoc2dial_dial_validation.json"}.items()}
    raw_docs = json.loads(sources["documents"].read_text(encoding="utf-8"))["doc_data"]
    raw_train = json.loads(sources["train"].read_text(encoding="utf-8"))["dial_data"]
    raw_val = json.loads(sources["validation"].read_text(encoding="utf-8"))["dial_data"]
    split = dialogue_split(raw_train, raw_val, seed=cfg["seed"], dev_fraction=cfg["validation_dev_fraction"])
    (destination / "splits.json").write_text(json.dumps(split, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    lookup = {item: role for role, ids in split["dialogues"].items() for item in ids}
    tokenizer = JsonTokenizer(tokenizer_path)
    chunks = []
    for domain, docs in raw_docs.items():
        for doc_id, doc in docs.items():
            values = chunk_document(domain, doc_id, doc["title"], doc["doc_text"], tokenizer,
                                    cfg["chunk_max_tokens"], cfg["chunk_overlap"], cfg["encoder_limit"])
            for c in values:
                if not c.text.endswith(doc["doc_text"][c.start:c.end]):
                    raise ValueError("chunk body differs from the raw character offsets")
            chunks.extend(values)
    with (destination / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for c in chunks:
            _line(handle, asdict(c))
    mapper = QrelsMapper(raw_docs, chunks, cfg["strict_min_span_coverage"])
    counts = {role: Counter() for role in split["dialogues"]}
    query_ids, samples, handles = set(), {k: [] for k in ("mapping_issue", "boundary", "multiple_refs", "ordinary")}, {}
    try:
        for role in split["dialogues"]:
            for kind in ("queries", "qrels"):
                handles[(role, kind)] = (destination / f"{kind}.{role}.jsonl").open(
                    "w", encoding="utf-8", newline="\n")
        with (destination / "qrels_issues.jsonl").open("w", encoding="utf-8", newline="\n") as issue_file:
            for source_role in ("train", "validation"):
                for example in examples(sources[source_role]):
                    qid = example["query_id"]
                    if qid in query_ids:
                        raise ValueError("duplicate query ID")
                    query_ids.add(qid)
                    role = lookup[example["dialogue_id"]]
                    valid = bool(example["question"].strip())
                    query = build_query(example["question"], example["history"], tokenizer,
                                        cfg["query_max_tokens"], history_turns=cfg["history_turns"]) if valid else Query("", 0, 0, False, False)
                    row = {"query_id": qid, "dialogue_id": example["dialogue_id"], "domain": example["domain"],
                           "question": example["question"].strip(), "query_text": query.text,
                           "valid_input": valid, "original_tokens": query.original_tokens,
                           "used_tokens": query.used_tokens, "truncated": query.truncated,
                           "current_question_truncated": query.current_question_truncated,
                           "history_turns": len(example["history"])}
                    _line(handles[(role, "queries")], row)
                    result = mapper.map(example["domain"], example["gold_references"])
                    _line(handles[(role, "qrels")], {"query_id": qid, "strict": result["strict"],
                        "lenient": result["lenient"], "references": len(result["references"])})
                    stats = counts[role]
                    stats.update(queries=1, invalid_inputs=int(not valid), truncated=int(query.truncated),
                                 current_question_truncated=int(query.current_question_truncated),
                                 with_strict=int(bool(result["strict"])), with_lenient=int(bool(result["lenient"])),
                                 references=len(result["references"]), reference_issues=len(result["issues"]))
                    for issue in result["issues"]:
                        _line(issue_file, {"query_id": qid, "split": role, **issue})
                        stats["issue_" + issue["issue"]] += 1
                    category = ("mapping_issue" if result["issues"] else
                                "boundary" if result["strict"] != result["lenient"] else
                                "multiple_refs" if len(result["references"]) > 1 else "ordinary")
                    if len(samples[category]) < 20:
                        refs = []
                        for r in result["references"]:
                            doc = raw_docs.get(example["domain"], {}).get(r["doc_id"], {})
                            refs.append(r | {"gold_excerpt": doc.get("doc_text", "")[r.get("start", 0):r.get("end", 0)][:400]})
                        samples[category].append({"query_id": qid, "split": role, "category": category,
                                                  "question": row["question"], "references": refs})
    finally:
        for handle in handles.values():
            handle.close()
    audit = []
    for i in range(20):
        for pool in samples.values():
            if i < len(pool) and len(audit) < 20:
                audit.append(pool[i])
    with (destination / "qrels_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in audit:
            _line(handle, row)
    retriever = BM25Chunks(chunks, tokenizer, query_max_tokens=cfg["query_max_tokens"],
                           dedup_overlap=cfg["dedup_overlap"], history_turns=cfg["history_turns"])
    names = ["chunks.jsonl", "splits.json", "qrels_issues.jsonl", "qrels_audit.jsonl"] + [
        f"{kind}.{role}.jsonl" for role in split["dialogues"] for kind in ("queries", "qrels")]
    files = [destination / name for name in names]
    manifest = {"corpus": "multidoc2dial", "artifact_version": retriever.version, "config": cfg,
        "relevance": {"strict": f"at least {cfg['strict_min_span_coverage']:.0%} of any one referenced span, half-open raw character offsets",
                      "lenient": "any positive raw character overlap", "multiple_refs": "union, not all-claims support"},
        "query_policy": "strip_current_reverse_history_newline_v1", "tokenizer_sha256": tokenizer.fingerprint,
        "sources_sha256": {k: digest(v) for k, v in sources.items()}, "documents": sum(map(len, raw_docs.values())),
        "chunks": len(chunks), "max_chunk_tokens": max(c.tokens for c in chunks),
        "dialogues": {k: len(v) for k, v in split["dialogues"].items()},
        "statistics": {k: dict(v) for k, v in counts.items()}, "audit_cases": len(audit),
        "files_sha256": {p.name: digest(p) for p in files}, "build_seconds": perf_counter() - started,
        "evaluation_status": "no heldout retrieval run; qrels construction is a data audit"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest
