"""Select a Docs-QA retrieval-risk threshold on DEV predictions only.

Input JSONL rows: {"split":"dev","score":0.5,"top5_ids":["x"],"gold_ids":["y"]}.
Use same retriever, score definition, chunk corpus and artifact as serving.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.evaluation import risk_coverage, select_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--method", choices=["R1", "R2", "R3", "R4", "R5"], required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--target-risk", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("artifacts/docs_gate.json"))
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.predictions.read_text(encoding="utf-8").splitlines() if s]
    if any(r["split"] != "dev" or not r["gold_ids"] for r in rows):
        raise ValueError("require dev rows with gold passages; unanswerable controls need separate labels")
    scores = [r["score"] for r in rows]
    errors = [not bool(set(r["top5_ids"][:5]) & set(r["gold_ids"])) for r in rows]
    gate = select_gate(scores, errors, method=args.method,
                       artifact_version=args.artifact_version, split="dev", target_risk=args.target_risk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"gate": asdict(gate), "risk_definition": "gold_not_in_top5",
                                    "curve": risk_coverage(scores, errors)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(gate)))


if __name__ == "__main__":
    main()
