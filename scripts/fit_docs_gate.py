"""Select a coverage operating point on DEV, with explicit score-policy binding."""

import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.evaluation import risk_coverage
from supportlens.docs_gate import select_coverage_gate
from supportlens.docs_eval import write_report, load_dev


def fit(rows, *, method, artifact_version=None, target_coverage=0.7, expected_qrels=None):
    if not rows or any(r.get("split") != "dev" for r in rows):
        raise ValueError("gate fitting requires dev-only input")
    selected = [r for r in rows if r.get("method") == method]
    if not selected or any(not r.get("gold_ids") for r in selected):
        raise ValueError("require selected-method dev rows with positive gold passages")
    versions, names = {r["artifact_version"] for r in selected}, {r["score_name"] for r in selected}
    expected = {"R1": "bm25_top1_per_term_v1", "R2": "dense_top1_cosine_v1",
                "R3": "max_dense_cosine_of_hybrid_top3_v1"}
    if (len(versions) != 1 or names != {expected[method]}
            or (artifact_version is not None and versions != {artifact_version})
            or len({r["query_id"] for r in selected}) != len(selected)):
        raise ValueError("mixed/duplicate artifacts or wrong confidence score policy")
    if expected_qrels is not None and (
            {r["query_id"] for r in selected} != set(expected_qrels)
            or any(set(r["gold_ids"]) != set(expected_qrels[r["query_id"]]["strict"]) for r in selected)):
        raise ValueError("require every eligible dev query once with frozen strict qrels; do not fit on smoke subsets")
    scores = [r["gate_score"] for r in selected]
    errors = [not bool(set(r["retrieved"][:3]) & set(r["gold_ids"])) for r in selected]
    gate = select_coverage_gate(scores, errors, method=method, artifact_version=next(iter(versions)),
                                split="dev", score_name=next(iter(names)), target_k=3,
                                target_coverage=target_coverage)
    curve = risk_coverage(scores, errors)
    accepted = sum(gate.accepts(score) for score in scores)
    point = next(p for p in curve if p["accepted"] == accepted) | {"threshold": gate.threshold}
    return {"gate": asdict(gate), "queries": len(selected), "operating_point": point,
            "risk_definition": "gold_not_in_top3_strict", "curve": curve,
            "threshold_placement": "midpoint between the last accepted and first rejected score groups when available",
            "caveat": "coverage target does not guarantee low risk; retrieval is not answer factuality; prompt may truncate evidence",
            "selection_status": "dev operating point; held-out remains unevaluated"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--method", choices=["R1", "R2", "R3"], required=True)
    parser.add_argument("--artifact-version")
    parser.add_argument("--target-coverage", type=float, default=0.7)
    parser.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/docs_gate.json"))
    args = parser.parse_args()
    payload = args.predictions.read_bytes()
    rows = [json.loads(s) for s in payload.decode("utf-8").splitlines() if s]
    _, queries, qrels = load_dev(args.bundle)
    expected_qrels = {q["query_id"]: qrels[q["query_id"]] for q in queries
                      if q["valid_input"] and qrels[q["query_id"]]["strict"]}
    report = fit(rows, method=args.method, artifact_version=args.artifact_version,
                 target_coverage=args.target_coverage, expected_qrels=expected_qrels)
    report["predictions_sha256"] = hashlib.sha256(payload).hexdigest()
    write_report(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}))


if __name__ == "__main__":
    main()
