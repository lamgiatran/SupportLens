"""Dev-only R1 check on the exact passage/query/dedup policy served by the API."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.retrieval import BM25Chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/docs_qa/dev_metrics.json"))
    parser.add_argument("--limit", type=int, help="smoke subset; never report it as the full dev run")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    for name in ("queries.dev.jsonl", "qrels.dev.jsonl", "splits.json"):
        actual = hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
        if actual != manifest["files_sha256"][name]:
            raise ValueError(f"bundle bytes changed: {name}")
    retriever = BM25Chunks.from_bundle(args.bundle, args.tokenizer)
    queries = [json.loads(line) for line in (args.bundle / "queries.dev.jsonl").read_text(encoding="utf-8").splitlines()]
    qrels = {r["query_id"]: r for r in map(json.loads, (args.bundle / "qrels.dev.jsonl").read_text(encoding="utf-8").splitlines())}
    queries = queries[:args.limit] if args.limit else queries
    records, times, values = [], [], {name: [] for name in ("strict", "lenient")}
    for row in queries:
        if not row["valid_input"]:
            continue
        started = perf_counter()
        evidence = retriever.search(row["query_text"], k=10)
        times.append((perf_counter() - started) * 1000)
        ids = [item.evidence_id for item in evidence]
        record = {"query_id": row["query_id"], "retrieved": ids}
        for relevance in values:
            gold = set(qrels[row["query_id"]][relevance])
            if not gold:
                record[relevance] = None
                continue
            hits = len(gold & set(ids[:5]))
            rr = next((1 / rank for rank, key in enumerate(ids, 1) if key in gold), 0.0)
            value = {"hit_at_5": float(hits > 0), "gold_recall_at_5": hits / len(gold), "mrr_at_10": rr}
            values[relevance].append(value)
            record[relevance] = value
        records.append(record)
    metrics = {}
    for relevance, rows in values.items():
        metrics[relevance] = {"evaluated_queries_with_positive_gold": len(rows),
            "queries_without_positive_gold": len(records) - len(rows),
            **{key: float(np.mean([r[key] for r in rows])) if rows else None
               for key in ("hit_at_5", "gold_recall_at_5", "mrr_at_10")}}
    report = {"split": "dev", "artifact_version": retriever.version,
        "scope": "smoke subset" if args.limit else "full dev, own token-bounded passages; not published-baseline reproduction",
        "queries_considered": len(queries), "valid_queries_retrieved": len(records), "metrics": metrics,
        "metric_definitions": {"hit_at_5": "at least one annotated positive in top-5",
                               "gold_recall_at_5": "fraction of all annotated gold chunks retrieved in top-5, macro over queries",
                               "mrr_at_10": "reciprocal rank of first positive within top-10"},
        "latency_ms": {"scope": "retrieval of prepared query only; one query per sample; warm BM25 index",
                       "note": "development run on a shared machine; not an isolated hardware or end-to-end benchmark",
                       "p50": float(np.percentile(times, 50)) if times else None,
                       "p95": float(np.percentile(times, 95)) if times else None},
        "unmapped_policy": "undefined gold queries excluded from that relevance metric and counted explicitly",
        "heldout_retrieval_run": False, "generation_run": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.out.with_name("dev_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
