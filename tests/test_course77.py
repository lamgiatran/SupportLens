import csv
import json
from pathlib import Path

import joblib

from supportlens.course77 import (
    PreprocessConfig,
    build_course77_splits,
    preprocess_text,
    train_and_evaluate_e1,
)


def test_preprocess_preserves_negation_and_lemmatizes():
    out = preprocess_text("I can't unblock my cards!\n", PreprocessConfig(remove_stopwords=True))
    assert "not" in out.split()
    assert "card" in out.split()
    assert "unblock" in out.split()


def test_build_course77_splits_and_e1_end_to_end(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    categories = [f"intent_{i:02d}" for i in range(77)]
    (raw_dir / "categories.json").write_text(json.dumps(categories), encoding="utf-8")

    with (raw_dir / "train.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "category"])
        for c in categories:
            for j in range(5):
                w.writerow([f"question {j} about {c} cards", c])
        # Internal duplicate & train-test overlap & blank row
        w.writerow(["\nquestion 0 about intent_00 cards\n", "intent_00"])
        w.writerow(["shared overlap question", "intent_01"])
        w.writerow(["   ", "intent_02"])

    with (raw_dir / "test.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "category"])
        for c in categories:
            w.writerow([f"official test query for {c}", c])
        w.writerow(["shared overlap question", "intent_01"])

    processed_dir = tmp_path / "processed"
    runs_dir = tmp_path / "runs"
    artifacts_dir = tmp_path / "artifacts"

    records, audit = build_course77_splits(
        raw_dir=raw_dir, processed_dir=processed_dir, runs_dir=runs_dir, dev_ratio=0.2, seed=42
    )
    assert audit["disjoint_normalized_text_verified"] is True
    assert audit["duplicates_and_overlap"]["train_test_normalized_overlap_count"] == 1
    assert audit["duplicates_and_overlap"]["train_internal_duplicate_extra_rows"] == 1
    assert audit["raw_counts"]["train_blanks"] == 1
    assert audit["split_counts"]["test"] == 78
    assert audit["split_counts"]["excluded"] == 3

    metrics = train_and_evaluate_e1(
        records,
        artifacts_dir=artifacts_dir,
        runs_dir=runs_dir,
        splits_path=processed_dir / "splits.json",
        seed=42,
    )
    assert metrics["official_test_touched"] is False
    assert (artifacts_dir / "e1_bow_lr.joblib").exists()
    pipe = joblib.load(artifacts_dir / "e1_bow_lr.joblib")
    assert len(pipe.predict(["question 0 about intent_00 cards"])) == 1
