"""Run Course-77 experiments: Package A (E1) and Package B (E1-E3 + lemmatization ablation)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.course77 import (
    build_course77_splits,
    load_course77_records_from_splits,
    run_package_b_suite,
    train_and_evaluate_e1,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["e1", "package_b"], default="package_b")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/course77"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/course77"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/course77"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--c-value", type=float, default=1.0)
    args = parser.parse_args()

    splits_path = args.processed_dir / "splits.json"
    if splits_path.exists():
        records, _ = load_course77_records_from_splits(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
        )
        audit_path = args.runs_dir / "course77_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    else:
        records, audit = build_course77_splits(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            runs_dir=args.runs_dir,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )

    if args.suite == "e1":
        metrics = train_and_evaluate_e1(
            records,
            artifacts_dir=args.artifacts_dir,
            runs_dir=args.runs_dir,
            splits_path=splits_path,
            c_value=args.c_value,
            seed=args.seed,
        )
        summary = {
            "split_counts": audit.get("split_counts"),
            "vocab_size": metrics["vocab_size"],
            "dev_metrics": metrics["metrics"],
            "error_count": metrics["error_count"],
            "smoke_examples": metrics["smoke_examples"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    summary = run_package_b_suite(
        records,
        artifacts_dir=args.artifacts_dir,
        runs_dir=args.runs_dir,
        splits_path=splits_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
