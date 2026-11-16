"""Compare confidence scores on DEV; AUROC label = gold retrieved in top-k."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.docs_eval import binary_auroc, write_report
from supportlens.evaluation import risk_coverage


def analyze(rows, *, k=3):
    if k < 1 or not rows or any(r.get("split") != "dev" or not r.get("gold_ids") for r in rows):
        raise ValueError("require nonempty dev rows with positive gold and positive k")
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("policy", row.get("method", "default"))].append(row)
    reports = {}
    for group, values in groups.items():
        if len({r["query_id"] for r in values}) != len(values):
            raise ValueError("duplicate query IDs within a policy")
        success = [bool(set(r["retrieved"][:k]) & set(r["gold_ids"])) for r in values]
        errors = [not s for s in success]
        scores = {}
        for name in ("raw_top1", "margin", "top1_per_term", "gate_score"):
            if all(name in r for r in values):
                s = [r[name] for r in values]
                scores[name] = {"auroc_success": binary_auroc(s, success),
                                "curve": risk_coverage(s, errors)}
        nterms, top = [r["nterms"] for r in values], [r["raw_top1"] for r in values]
        corr = float(np.corrcoef(nterms, top)[0, 1]) if np.std(nterms) > 0 and np.std(top) > 0 else None
        reports[group] = {"queries": len(values), "unconditional_risk": float(np.mean(errors)),
                          "raw_top1_query_terms_pearson": corr, "scores": scores}
    return {"split": "dev", "target_k": k, "risk_definition": f"gold_not_in_top{k}",
            "scope": "retrieval proxy; not answer correctness, not unanswerability detection",
            "groups": reports, "heldout_retrieval_run": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", type=Path)
    p.add_argument("--k", type=int, choices=[3, 5], default=3)
    p.add_argument("--out", type=Path, default=Path("runs/docs_qa/gate_scores_metrics.json"))
    a = p.parse_args()
    payload = a.predictions.read_bytes()
    report = analyze([json.loads(line) for line in payload.decode("utf-8").splitlines() if line], k=a.k)
    report["predictions_sha256"] = hashlib.sha256(payload).hexdigest()
    write_report(a.out, report)
    # Full tie-aware curves are in the file; keep console output compact.
    print(json.dumps({g: {s: v["auroc_success"] for s, v in r["scores"].items()}
                      for g, r in report["groups"].items()}))
