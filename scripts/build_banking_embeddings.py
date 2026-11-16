"""One command to create the cache Cau can consume with numpy only."""

import argparse
import json
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.banking import BankingEncoder, EncoderConfig
from supportlens.banking_data import build_cache, load_handoff_rows, write_split_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/banking77_oos"))
    parser.add_argument("--config", type=Path, default=Path("configs/banking_encoder.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/handoff50"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="validate sources without loading a model")
    args = parser.parse_args()
    rows, hashes = load_handoff_rows(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    write_split_manifest(args.out / "splits.json", {
        "model_scope": "handoff50", "source_sha256": hashes,
        "samples": [{k: v for k, v in r.items() if k != "text"} for r in rows]
    })
    audit = {"rows": len(rows), "fit_rows": sum(r["fit_eligible"] for r in rows),
             "invalid_input_ids": [r["sample_id"] for r in rows if not r["valid_input"]],
             "labels": len({r["label"] for r in rows if r["fit_eligible"]}),
             "source_files": len(hashes),
             "counts": dict(Counter(r["split"] + "/" + r["group"] for r in rows))}
    audit_path = Path("runs/handoff50/data_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if args.audit_only:
        print(json.dumps(audit))
        return 0
    try:
        encoder = BankingEncoder(EncoderConfig.load(args.config), device=args.device,
                                 local_files_only=args.local_files_only)
        report = build_cache(rows, encoder, args.out, source_hashes=hashes,
                             batch_size=args.batch_size)
    except (ImportError, OSError) as error:
        print(json.dumps({"status": "unavailable", "error_type": type(error).__name__,
                          "message": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
