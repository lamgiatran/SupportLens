"""Run Course-77 Package A: audit BANKING77, build train_fit/dev splits, and run E1."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.course77 import build_course77_splits, train_and_evaluate_e1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/course77"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/course77"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/course77"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--c-value", type=float, default=1.0)
    args = parser.parse_args()

    records, audit = build_course77_splits(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        runs_dir=args.runs_dir,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    metrics = train_and_evaluate_e1(
        records,
        artifacts_dir=args.artifacts_dir,
        runs_dir=args.runs_dir,
        splits_path=args.processed_dir / "splits.json",
        c_value=args.c_value,
        seed=args.seed,
    )
    summary = {
        "split_counts": audit["split_counts"],
        "train_test_overlaps": audit["duplicates_and_overlap"]["train_test_normalized_overlap_count"],
        "train_internal_duplicate_extra_rows": audit["duplicates_and_overlap"]["train_internal_duplicate_extra_rows"],
        "vocab_size": metrics["vocab_size"],
        "dev_metrics": metrics["metrics"],
        "error_count": metrics["error_count"],
        "smoke_examples": metrics["smoke_examples"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
