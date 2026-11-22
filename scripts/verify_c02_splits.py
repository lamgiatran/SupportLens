"""C-02 Calibration Split & Disjointness Verification Script.

Owner: Nguyen Huu Cau (2352129)
Verifies all C-02 invariants on data/processed/handoff50/splits.json:
1. Seed-42 determinism (re-running assign_calibration_split produces identical masks).
2. Exact 1-to-1 sample_id order preservation across all 12,221 rows.
3. Strict disjointness by sample_id AND by real MiniLM token_ids:
   - fit_eligible vs calibration
   - fit_eligible vs dev / test / id_oos / ood / invalid
   - calibration vs dev / test / id_oos / ood / invalid
   - dev_eligible vs test
4. Internal duplicate group integrity (id/train/3389 & 3401 kept in same partition).
5. All 50 in-scope labels present in both fit and calibration; per-label counts logged.
6. Exports audit report to runs/handoff50/calibration_audit.json.
"""

import hashlib
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer as HFTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.banking_data import (
    EXCLUDED_DEV_DUPLICATES,
    EXCLUDED_TRAIN_LEAKS,
    assign_calibration_split,
    load_handoff_rows,
)


def main():
    raw_root = Path("data/raw/banking77_oos")
    splits_path = Path("data/processed/handoff50/splits.json")
    tok_path = Path("data/tokenizers/minilm/tokenizer.json")
    manifest_path = Path("data/processed/handoff50/embedding_manifest.json")

    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}. Run build_banking_embeddings.py --audit-only first.")
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing {tok_path}. Run fetch_tokenizers.py first.")

    saved = json.loads(splits_path.read_text(encoding="utf-8"))
    saved_samples = saved["samples"]

    # 1. Re-compute deterministically from raw + seed=42
    fresh_rows, _ = load_handoff_rows(raw_root)
    cal_summary = assign_calibration_split(fresh_rows, fraction=0.10, seed=42, tokenizer_path=tok_path)

    assert len(saved_samples) == len(fresh_rows), "row count mismatch"
    for s_row, f_row in zip(saved_samples, fresh_rows):
        assert s_row["sample_id"] == f_row["sample_id"], f"order/ID mismatch: {s_row['sample_id']} vs {f_row['sample_id']}"
        assert s_row["label"] == f_row["label"]
        assert s_row["fit_eligible"] == f_row["fit_eligible"]
        assert s_row["calibration"] == f_row["calibration"]
        assert s_row["valid_input"] == f_row["valid_input"]
        assert s_row["dev_eligible"] == f_row["dev_eligible"]

    # 2. Tokenize all valid rows with pinned MiniLM tokenizer
    tok = HFTokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=256)
    tok.no_padding()
    valid_texts = [r["text"].strip() for r in fresh_rows]
    encodings = tok.encode_batch(valid_texts, add_special_tokens=True)
    tok_by_id = {
        r["sample_id"]: tuple(enc.ids)
        for r, enc in zip(fresh_rows, encodings)
        if r["valid_input"]
    }

    fit_ids = {r["sample_id"] for r in fresh_rows if r["fit_eligible"]}
    cal_ids = {r["sample_id"] for r in fresh_rows if r["calibration"]}
    dev_all_ids = {r["sample_id"] for r in fresh_rows if r["split"] == "dev" and r["valid_input"]}
    dev_elig_ids = {r["sample_id"] for r in fresh_rows if r["dev_eligible"]}
    test_ids = {r["sample_id"] for r in fresh_rows if r["split"] == "test" and r["valid_input"]}
    oos_ids = {r["sample_id"] for r in fresh_rows if r["group"] in {"id_oos", "ood"}}
    invalid_ids = {r["sample_id"] for r in fresh_rows if not r["valid_input"]}

    # 3. Sample-ID disjointness
    assert not (fit_ids & cal_ids), "fit and calibration overlap by sample_id"
    assert not (fit_ids & dev_all_ids), "fit and dev overlap by sample_id"
    assert not (fit_ids & test_ids), "fit and test overlap by sample_id"
    assert not (cal_ids & dev_all_ids), "calibration and dev overlap by sample_id"
    assert not (cal_ids & test_ids), "calibration and test overlap by sample_id"
    assert not ((fit_ids | cal_ids) & oos_ids), "OOS rows leaked into fit or calibration"
    assert not ((fit_ids | cal_ids) & invalid_ids), "invalid blank rows in fit or calibration"
    assert not ((fit_ids | cal_ids) & EXCLUDED_TRAIN_LEAKS), "C-01 excluded train leaks in fit or calibration"
    assert not (dev_elig_ids & EXCLUDED_DEV_DUPLICATES), "C-01 excluded dev duplicates in dev_eligible"

    # 4. MiniLM Token-ID disjointness
    fit_toks = {tok_by_id[i] for i in fit_ids}
    cal_toks = {tok_by_id[i] for i in cal_ids}
    dev_all_toks = {tok_by_id[i] for i in dev_all_ids}
    dev_elig_toks = {tok_by_id[i] for i in dev_elig_ids}
    test_toks = {tok_by_id[i] for i in test_ids}

    assert not (fit_toks & cal_toks), "fit and calibration share MiniLM token_ids!"
    assert not (fit_toks & dev_all_toks), "fit and dev share MiniLM token_ids!"
    assert not (fit_toks & test_toks), "fit and test share MiniLM token_ids!"
    assert not (cal_toks & dev_all_toks), "calibration and dev share MiniLM token_ids!"
    assert not (cal_toks & test_toks), "calibration and test share MiniLM token_ids!"
    assert not (dev_elig_toks & test_toks), "dev_eligible and test share MiniLM token_ids!"

    # 5. Check id/train/3389 & id/train/3401 kept together
    row_3389 = next(r for r in fresh_rows if r["sample_id"] == "id/train/3389")
    row_3401 = next(r for r in fresh_rows if r["sample_id"] == "id/train/3401")
    assert (row_3389["fit_eligible"] == row_3401["fit_eligible"]) and (
        row_3389["calibration"] == row_3401["calibration"]
    ), "internal duplicate pair 3389/3401 was split!"

    # 6. Check 50 labels
    assert cal_summary["labels_in_fit"] == 50
    assert cal_summary["labels_in_calibration"] == 50

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    report = {
        "status": "pass",
        "owner": "Nguyen Huu Cau (2352129)",
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "encoder_fingerprint": manifest.get("encoder_fingerprint"),
        "npz_exists_locally": Path("data/processed/handoff50/embeddings.npz").exists(),
        "invariants_checked": {
            "seed_42_reproducible": True,
            "row_order_preserved_12221": True,
            "sample_id_disjoint": True,
            "minilm_token_id_disjoint_fit_cal_dev_test": True,
            "dev_eligible_vs_test_token_id_disjoint": True,
            "internal_train_dup_3389_3401_kept_together": True,
            "oos_and_invalid_excluded_from_fit_and_cal": True,
            "all_50_labels_in_fit_and_cal": True,
        },
        "counts": {
            "total_rows": len(fresh_rows),
            "fit_eligible": len(fit_ids),
            "calibration": len(cal_ids),
            "excluded_train_leaks": len(EXCLUDED_TRAIN_LEAKS),
            "excluded_dev_duplicates": len(EXCLUDED_DEV_DUPLICATES),
            "dev_eligible": len(dev_elig_ids),
            "test_valid": len(test_ids),
        },
        "calibration_summary": cal_summary,
    }
    out_path = Path("runs/handoff50/calibration_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "calibration_summary"}, indent=2))
    print(f"fit={len(fit_ids)}, calibration={len(cal_ids)}, fraction={cal_summary['actual_fraction']:.4f}")
    print(f"3389 & 3401 partition: {'calibration' if row_3389['calibration'] else 'fit'}")


if __name__ == "__main__":
    main()
