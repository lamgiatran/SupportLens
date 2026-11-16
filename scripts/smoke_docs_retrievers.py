"""Real-encoder API/offline parity on dev queries, including gate boundaries."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.api import app_from_env
from supportlens.docs_eval import load_dev, write_report
from supportlens.docs_gate import DocsGate
from supportlens.retrieval import BM25Chunks


def run(bundle, tokenizer, config, cache, gate_dir, raw_validation, predictions, out):
    _, queries, _ = load_dev(bundle, raw_validation=raw_validation)
    base = BM25Chunks.from_bundle(bundle, tokenizer)
    methods = ("R1", "R2", "R3")
    gates = {m: DocsGate(**json.loads((Path(gate_dir) / (m + ".json")).read_text(encoding="utf-8"))["gate"])
             for m in methods}
    rows = [json.loads(line) for line in Path(predictions).read_text(encoding="utf-8").splitlines() if line]
    allowed = {q["query_id"] for q in queries}
    if any(r.get("split") != "dev" or r["query_id"] not in allowed for r in rows):
        raise ValueError("require dev-only offline predictions")
    by_prediction = {(r["method"], r["query_id"]): r for r in rows}
    selected = set()
    for domain in sorted({q["domain"] for q in queries}):
        group = [q for q in queries if q["domain"] == domain]
        candidates = [max(group, key=lambda q: len(q["history"])), min(group, key=lambda q: len(q["history"]))] + group
        chosen = []
        for q in candidates:
            if q["query_id"] not in chosen:
                chosen.append(q["query_id"])
            if len(chosen) == 5:
                break
        selected.update(chosen)
    # Include both sides of each operating point to catch batch/single drift.
    for method, gate in gates.items():
        candidates = sorted((r for r in rows if r["method"] == method),
                            key=lambda r: abs(r["gate_score"] - gate.threshold))
        selected.update(r["query_id"] for r in candidates[:2])
    samples, errors, latencies = [], {m: [] for m in methods}, {m: [] for m in methods}
    with patch.dict(os.environ, {"SUPPORTLENS_DOCS_BUNDLE": str(bundle),
        "SUPPORTLENS_DOCS_TOKENIZER": str(tokenizer), "SUPPORTLENS_DOCS_ENCODER": str(config),
        "SUPPORTLENS_DOCS_DENSE": str(cache), "SUPPORTLENS_DOCS_GATES": str(gate_dir),
        "SUPPORTLENS_DOCS_GATE": "", "SUPPORTLENS_ENCODER_DEVICE": "cpu",
        "SUPPORTLENS_LOG": str(Path(out).with_suffix(".requests.jsonl")), "SUPPORTLENS_LOG_QUESTIONS": "0"}):
        app = app_from_env()
    with TestClient(app) as client:
        health = client.get("/health").json()
        if not health["docs_encoder_ready"]:
            raise ValueError("encoder did not warm up before requests")
        for row in queries:
            if row["query_id"] not in selected:
                continue
            if base.prepare_query(row["question"], row["history"]).text != row["query_text"]:
                raise ValueError("API/offline packed query differs")
            for method in methods:
                expected = by_prediction[method, row["query_id"]]
                response = client.post("/v1/assist", json={"mode": "docs_qa", "retriever": method,
                    "question": row["question"], "history": row["history"]})
                response.raise_for_status()
                value = response.json()
                expected_decision = "retrieved" if gates[method].accepts(expected["gate_score"]) else "handoff"
                difference = abs(value["score"] - expected["gate_score"])
                if (value["artifact_version"] != expected["artifact_version"]
                        or [e["evidence_id"] for e in value["evidence"]] != expected["retrieved"][:5]
                        or difference > 1e-5 or value["decision"] != expected_decision):
                    raise ValueError(f"API/offline mismatch: {method} {row['query_id']} score delta={difference}")
                errors[method].append(difference)
                latencies[method].append(value["latency"]["retrieve_ms"])
                samples.append({"query_id": row["query_id"], "domain": row["domain"], "method": method,
                                "decision": value["decision"], "score_abs_delta": difference})
    report = {"status": "pass", "split": "dev", "queries": len(selected), "requests": len(samples),
        "scope": "in-process API smoke with real per-request encoder; not a latency benchmark or generation evaluation",
        "startup": health, "all_top5_and_decisions_match_offline": True,
        "maximum_score_abs_difference": {m: max(v) for m, v in errors.items()},
        "sample_retrieve_ms_including_query_encode": {m: {"median": float(np.median(v)), "max": max(v)}
                                                      for m, v in latencies.items()},
        "decisions": dict(Counter(s["decision"] for s in samples)), "samples": samples,
        "heldout_retrieval_run": False, "generation_run": False}
    write_report(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("data/processed/docs_qa"))
    p.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    p.add_argument("--encoder-config", type=Path, default=Path("configs/docs_encoder.json"))
    p.add_argument("--cache", type=Path, default=Path("data/processed/docs_dense"))
    p.add_argument("--gate-dir", type=Path, default=Path("artifacts/docs_gates"))
    p.add_argument("--raw-validation", type=Path, default=Path("data/raw/multidoc2dial/multidoc2dial_dial_validation.json"))
    p.add_argument("--predictions", type=Path, default=Path("runs/docs_qa/retrieval_predictions.dev.jsonl"))
    p.add_argument("--out", type=Path, default=Path("runs/docs_qa/retrievers_api_audit.json"))
    a = p.parse_args()
    run(a.bundle, a.tokenizer, a.encoder_config, a.cache, a.gate_dir, a.raw_validation, a.predictions, a.out)
