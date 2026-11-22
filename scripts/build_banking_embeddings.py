"""One command to create the cache Cau can consume with numpy only."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.banking import BankingEncoder, EncoderConfig
from supportlens.banking_data import (
    EXCLUDED_DEV_DUPLICATES,
    EXCLUDED_TRAIN_LEAKS,
    assign_calibration_split,
    build_cache,
    load_handoff_rows,
    write_split_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/banking77_oos"))
    parser.add_argument("--config", type=Path, default=Path("configs/banking_encoder.json"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizers/minilm/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/handoff50"))
    parser.add_argument("--cal-fraction", type=float, default=0.10)
    parser.add_argument("--cal-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="validate sources and write splits without loading encoder")
    args = parser.parse_args()

    rows, hashes = load_handoff_rows(args.raw)
    cal_summary = assign_calibration_split(
        rows,
        fraction=args.cal_fraction,
        seed=args.cal_seed,
        tokenizer_path=args.tokenizer,
    )
    calibration_policy = {
        "seed": args.cal_seed,
        "target_fraction": args.cal_fraction,
        "actual_fraction": round(cal_summary["actual_fraction"], 6),
        "fit_rows": cal_summary["fit_rows"],
        "calibration_rows": cal_summary["calibration_rows"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_split_manifest(args.out / "splits.json", {
        "model_scope": "handoff50",
        "calibration_policy": calibration_policy,
        "source_sha256": hashes,
        "samples": [{k: v for k, v in r.items() if k != "text"} for r in rows],
    })
    audit = {
        "rows": len(rows),
        "fit_rows": sum(r["fit_eligible"] for r in rows),
        "calibration_rows": sum(r["calibration"] for r in rows),
        "invalid_input_ids": [r["sample_id"] for r in rows if not r["valid_input"]],
        "excluded_train_leak_ids": sorted(EXCLUDED_TRAIN_LEAKS),
        "excluded_dev_duplicate_ids": sorted(EXCLUDED_DEV_DUPLICATES),
        "labels": len({r["label"] for r in rows if r["fit_eligible"]}),
        "source_files": len(hashes),
        "counts": dict(Counter(r["split"] + "/" + r["group"] for r in rows)),
        "calibration_summary": cal_summary,
    }
    audit_path = Path("runs/handoff50/data_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        print(json.dumps({k: v for k, v in audit.items() if k != "calibration_summary"}))
        return 0
    try:
        encoder = BankingEncoder(EncoderConfig.load(args.config), device=args.device,
                                 local_files_only=args.local_files_only)
        report = build_cache(rows, encoder, args.out, source_hashes=hashes,
                             batch_size=args.batch_size,
                             calibration_policy=calibration_policy)
    except (ImportError, OSError) as error:
        print(json.dumps({"status": "unavailable", "error_type": type(error).__name__,
                          "message": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
