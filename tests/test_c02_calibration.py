"""Unit tests for C-02 calibration partitioning, group stratification, and NPZ masks."""

import json
import numpy as np
import pytest

from supportlens.banking import BankingEncoder, EncoderConfig
from supportlens.banking_data import assign_calibration_split, build_cache


class _TestEncoder:
    def encode(self, texts, **kwargs):
        arr = np.asarray([[1.0, float(len(t)), 2.0, 3.0] for t in texts], dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def _make_rows():
    rows = []
    for label_idx in range(50):
        label = f"intent_{label_idx:02d}"
        for j in range(10):
            # Duplicate text on intent_00 indices 0 and 1 (different spacing)
            if label_idx == 0 and j in (0, 1):
                text = "why is my card declined?" if j == 0 else "why is my card  declined?"
            else:
                text = f"question {label_idx} item {j}"
            rows.append({
                "sample_id": f"id/train/{label_idx * 10 + j}",
                "text": text,
                "label": label,
                "original_intent": label,
                "split": "train",
                "group": "id",
                "valid_input": True,
                "calibration": False,
                "fit_eligible": True,
                "dev_eligible": False,
            })
    # Add an excluded leak row and an OOS row
    rows.append({
        "sample_id": "id/train/9999",
        "text": "leaked train row",
        "label": "intent_00",
        "original_intent": "intent_00",
        "split": "train",
        "group": "id",
        "valid_input": True,
        "calibration": False,
        "fit_eligible": False,
        "dev_eligible": False,
    })
    rows.append({
        "sample_id": "id_oos/valid/0",
        "text": "oos question",
        "label": "oos",
        "original_intent": "withheld_intent",
        "split": "dev",
        "group": "id_oos",
        "valid_input": True,
        "calibration": False,
        "fit_eligible": False,
        "dev_eligible": True,
    })
    return rows


def test_calibration_split_is_deterministic_group_stratified_and_disjoint():
    rows_a = _make_rows()
    rows_b = _make_rows()
    summary_a = assign_calibration_split(rows_a, fraction=0.10, seed=42, tokenizer_path=None)
    summary_b = assign_calibration_split(rows_b, fraction=0.10, seed=42, tokenizer_path=None)

    assert [r["calibration"] for r in rows_a] == [r["calibration"] for r in rows_b]
    assert [r["fit_eligible"] for r in rows_a] == [r["fit_eligible"] for r in rows_b]
    assert summary_a["labels_in_fit"] == 50
    assert summary_a["calibration_rows"] >= 45
    assert not any(r["fit_eligible"] and r["calibration"] for r in rows_a)
    # Excluded leak and OOS rows never enter fit or calibration
    assert not rows_a[-2]["fit_eligible"] and not rows_a[-2]["calibration"]
    assert not rows_a[-1]["fit_eligible"] and not rows_a[-1]["calibration"]
    # Duplicate group (id/train/0 and id/train/1) stays in the exact same partition
    assert rows_a[0]["calibration"] == rows_a[1]["calibration"]
    assert rows_a[0]["fit_eligible"] == rows_a[1]["fit_eligible"]


def test_build_cache_writes_calibration_bool_array_and_preserves_order(tmp_path):
    rows = _make_rows()
    summary = assign_calibration_split(rows, fraction=0.10, seed=42, tokenizer_path=None)
    policy = {"seed": 42, "target_fraction": 0.10, "fit_rows": summary["fit_rows"], "calibration_rows": summary["calibration_rows"]}
    enc = BankingEncoder(EncoderConfig("offline-test", "b" * 40, dimension=4), backend=_TestEncoder())
    manifest = build_cache(rows, enc, tmp_path, source_hashes={}, calibration_policy=policy)
    with np.load(tmp_path / "embeddings.npz", allow_pickle=False) as cache:
        assert "calibration" in cache
        assert cache["calibration"].dtype == np.bool_
        assert cache["calibration"].tolist() == [r["calibration"] for r in rows]
        assert cache["fit_eligible"].tolist() == [r["fit_eligible"] for r in rows]
        assert cache["sample_id"].tolist() == [r["sample_id"] for r in rows]
        assert not (cache["fit_eligible"] & cache["calibration"]).any()
    splits_saved = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    assert splits_saved["calibration_policy"] == policy
    assert manifest["calibration_rows"] == sum(r["calibration"] for r in rows)
    assert manifest["fit_rows"] == sum(r["fit_eligible"] for r in rows)


def test_missing_explicit_tokenizer_raises_file_not_found(tmp_path):
    rows = _make_rows()
    with pytest.raises(FileNotFoundError, match="Missing pinned MiniLM tokenizer"):
        assign_calibration_split(rows, fraction=0.10, seed=42, tokenizer_path=tmp_path / "nonexistent.json")
