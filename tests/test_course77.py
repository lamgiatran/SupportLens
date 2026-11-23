import builtins
import csv
import json
import sys
from types import SimpleNamespace

import pytest

from supportlens.course77 import (
    Course77RunSpec,
    PreprocessConfig,
    build_course77_splits,
    load_course77_records_from_splits,
    preprocess_text,
    run_package_b_suite,
    train_and_evaluate_e1,
)


def _has_wordnet() -> bool:
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        return True
    except (ImportError, LookupError):
        return False


def test_preprocess_without_lemmatize_works_in_dev_only():
    out = preprocess_text(
        "I can't unblock my cards!\n",
        PreprocessConfig(lemmatize=False, remove_stopwords=True),
    )
    assert "not" in out.split()
    assert "cards" in out.split()
    assert "unblock" in out.split()


def test_preprocess_missing_nltk_raises_clear_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "nltk" or name.startswith("nltk."):
            raise ImportError("mocked missing nltk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "nltk", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"\[classical\]"):
        preprocess_text("cards", PreprocessConfig(lemmatize=True))


def test_preprocess_missing_wordnet_raises_clear_error(monkeypatch):
    nltk_corpus = pytest.importorskip("nltk.corpus")

    def fake_ensure_loaded():
        raise LookupError("mocked missing wordnet")

    monkeypatch.setattr(nltk_corpus, "wordnet", SimpleNamespace(ensure_loaded=fake_ensure_loaded))
    with pytest.raises(LookupError, match=r"nltk\.download\('wordnet'\)"):
        preprocess_text("cards", PreprocessConfig(lemmatize=True))


def test_preprocess_preserves_negation_and_lemmatizes():
    pytest.importorskip("nltk")
    if not _has_wordnet():
        pytest.skip("WordNet corpus not downloaded; run nltk.download('wordnet')")
    out = preprocess_text("I can't unblock my cards!\n", PreprocessConfig(remove_stopwords=True))
    assert "not" in out.split()
    assert "card" in out.split()
    assert "unblock" in out.split()


def test_build_course77_splits_and_package_b_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")

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

    loaded_records, _ = load_course77_records_from_splits(raw_dir=raw_dir, processed_dir=processed_dir)
    assert len(loaded_records) == len(records)

    use_lemma = _has_wordnet()
    e1_metrics = train_and_evaluate_e1(
        loaded_records,
        artifacts_dir=artifacts_dir,
        runs_dir=runs_dir,
        splits_path=processed_dir / "splits.json",
        preprocess_config=PreprocessConfig(lemmatize=use_lemma),
        seed=42,
    )
    assert e1_metrics["official_test_touched"] is False

    specs = (
        Course77RunSpec(
            experiment_id="E1",
            slug="e1_bow_lr",
            metrics_filename="e1_dev_metrics.json",
            predictions_filename="e1_dev_predictions.jsonl",
            representation="bow",
            classifier_type="logistic_regression",
            preprocess=PreprocessConfig(lemmatize=use_lemma),
        ),
        Course77RunSpec(
            experiment_id="E2",
            slug="e2_tfidf_lr",
            metrics_filename="e2_dev_metrics.json",
            predictions_filename="e2_dev_predictions.jsonl",
            representation="tfidf",
            classifier_type="logistic_regression",
            preprocess=PreprocessConfig(lemmatize=use_lemma),
        ),
        Course77RunSpec(
            experiment_id="E3",
            slug="e3_tfidf_svm",
            metrics_filename="e3_dev_metrics.json",
            predictions_filename="e3_dev_predictions.jsonl",
            representation="tfidf",
            classifier_type="linear_svm",
            preprocess=PreprocessConfig(lemmatize=use_lemma),
        ),
        Course77RunSpec(
            experiment_id="E2_no_lemma",
            slug="e2_tfidf_lr_no_lemma",
            metrics_filename="e2_no_lemma_dev_metrics.json",
            predictions_filename="e2_no_lemma_dev_predictions.jsonl",
            representation="tfidf",
            classifier_type="logistic_regression",
            preprocess=PreprocessConfig(lemmatize=False),
        ),
    )
    summary = run_package_b_suite(
        loaded_records,
        artifacts_dir=artifacts_dir,
        runs_dir=runs_dir,
        splits_path=processed_dir / "splits.json",
        specs=specs,
    )
    assert [row["experiment_id"] for row in summary["comparison_table"]] == ["E1", "E2", "E3", "E2_no_lemma"]
    assert (runs_dir / "package_b_comparison_metrics.json").exists()
    pipe = joblib.load(artifacts_dir / "e3_tfidf_svm.joblib")
    assert len(pipe.predict(["question 0 about intent_00 cards"])) == 1
