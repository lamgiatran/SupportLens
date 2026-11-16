"""Reproduce 0/1/2/all-history R1 experiments on DEV only (one turn = one message)."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.docs_eval import load_dev, query_metrics, aggregate_metrics, paired_dialogue_bootstrap, write_report
from supportlens.retrieval import BM25Chunks, words


def run(bundle, tokenizer, raw_validation, out, *, split="dev", limit=None, bootstrap=2000):
    if split != "dev":
        raise ValueError("history selection is dev-only")
    manifest, queries, qrels = load_dev(bundle, raw_validation=raw_validation)
    queries = [q for q in queries if q["valid_input"]]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        queries = queries[:limit]
    base = BM25Chunks.from_bundle(bundle, tokenizer)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports, paired = {}, {}
    records_path = output.with_name("history_predictions.jsonl")
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for label, turns in [("current", 0), ("last1", 1), ("last2", 2), ("all", None)]:
            retriever = BM25Chunks(base.chunks, base.tokenizer, query_max_tokens=base.query_max_tokens,
                                    dedup_overlap=base.dedup_overlap, history_turns=turns)
            values = {name: [] for name in ("strict", "lenient")}
            lengths, truncations = [], 0
            started = perf_counter()
            for row in queries:
                query = retriever.prepare_query(row["question"], row["history"])
                result = retriever.search(query.text, k=10)
                ids = [r.evidence_id for r in result]
                labels = qrels[row["query_id"]]
                for relevance in values:
                    values[relevance].append(query_metrics(ids, labels[relevance]))
                nterms = len(words(query.text))
                record = {"split": "dev", "policy": label, "history_turns": turns,
                          "query_id": row["query_id"], "dialogue_id": row["dialogue_id"],
                          "artifact_version": retriever.version, "retrieved": ids,
                          "gold_ids": labels["strict"], "nterms": nterms,
                          "raw_top1": result[0].score, "margin": result[0].score - result[1].score,
                          "top1_per_term": result[0].score / max(1, nterms)}
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                lengths.append(query.used_tokens)
                truncations += int(query.truncated)
            reports[label] = {"history_turns": turns, "artifact_version": retriever.version,
                              "metrics": {k: aggregate_metrics(v) for k, v in values.items()},
                              "query_tokens_mean": float(np.mean(lengths)), "truncated": truncations,
                              "seconds": perf_counter() - started}
            paired[label] = values["strict"]
            print(json.dumps({"policy": label, **reports[label]}), flush=True)
    valid = [i for i in range(len(queries)) if paired["last2"][i] is not None and paired["all"][i] is not None]
    report = {"split": "dev", "scope": "smoke subset" if limit else "full dev", "queries": len(queries),
              "one_turn_means": "one preceding message (user or assistant), not a user-agent exchange",
              "input_hashes": {"raw_validation": manifest["sources_sha256"]["validation"],
                               "chunks": manifest["files_sha256"]["chunks.jsonl"],
                               "qrels_dev": manifest["files_sha256"]["qrels.dev.jsonl"]},
              "policies": reports, "last2_minus_all_mrr10": paired_dialogue_bootstrap(
                  [queries[i]["dialogue_id"] for i in valid],
                  [paired["last2"][i]["mrr_at_10"] for i in valid],
                  [paired["all"][i]["mrr_at_10"] for i in valid], repeats=bootstrap),
              "heldout_retrieval_run": False, "generation_run": False}
    write_report(output, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    p.add_argument("--raw-validation", type=Path, default=Path("data/raw/multidoc2dial/multidoc2dial_dial_validation.json"))
    p.add_argument("--out", type=Path, default=Path("runs/docs_qa/history_metrics.json"))
    p.add_argument("--split", choices=["dev"], default="dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    run(a.bundle, a.tokenizer, a.raw_validation, a.out, split=a.split, limit=a.limit, bootstrap=a.bootstrap)
