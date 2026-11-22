"""Course-77 (BANKING77) audit, stratified split, preprocessing, and E1-E3 baselines.

This module satisfies the NLP course experiment track (K5) independently of the
50-label Handoff-50 application pipeline. Vocabulary and classifier parameters
are strictly fitted on train_fit only; dev selects configuration; official test
remains untouched until final freeze in Package D.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
import unicodedata

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class TransformerMixin:  # type: ignore[no-redef]
        pass

from .banking_data import write_split_manifest

NEGATION_WORDS = frozenset({
    "no", "not", "nor", "never", "neither", "cannot",
    "nothing", "nowhere", "none", "without",
})

# Keep core interrogatives/modals important for banking intent disambiguation.
LIGHT_STOPWORDS = frozenset({
    "a", "an", "the", "am", "is", "are", "was", "were", "be", "been", "being",
    "of", "at", "by", "for", "with", "about", "into", "through", "to", "from",
    "in", "on", "that", "this", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "they", "their", "and", "or", "but", "so", "if",
    "as", "just", "very", "really", "please", "hi", "hello",
}) - NEGATION_WORDS

_CONTRACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), " is"),
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _require_classical():
    try:
        import joblib
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise ImportError(
            "Course-77 training and splitting require classical ML dependencies. "
            "Install them with: pip install -e \".[classical]\""
        ) from exc
    return {
        "joblib": joblib,
        "CountVectorizer": CountVectorizer,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "LinearSVC": LinearSVC,
        "StratifiedShuffleSplit": StratifiedShuffleSplit,
        "Pipeline": Pipeline,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
    }


def normalize_for_dedup(text: str) -> str:
    return text.strip().casefold()


@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    expand_contractions: bool = True
    lemmatize: bool = True
    remove_stopwords: bool = False
    keep_negation: bool = True


def _get_wordnet_lemmatizer():
    try:
        from nltk.corpus import wordnet as wn
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise ImportError(
            "Lemmatization is enabled but nltk is not installed. "
            "Install classical extras with: pip install -e \".[classical]\""
        ) from exc

    try:
        wn.ensure_loaded()
    except LookupError as exc:
        raise LookupError(
            "Lemmatization is enabled but NLTK WordNet corpus is missing. "
            "Download it with: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        ) from exc

    return WordNetLemmatizer()


def _lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmatizer = _get_wordnet_lemmatizer()
    try:
        # Lemmatize verb then noun so plural nouns and inflected verbs collapse cleanly.
        return [lemmatizer.lemmatize(lemmatizer.lemmatize(tok, pos="v"), pos="n") for tok in tokens]
    except LookupError as exc:
        raise LookupError(
            "Lemmatization is enabled but NLTK WordNet data (wordnet / omw-1.4) is incomplete. "
            "Download it with: python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        ) from exc


def preprocess_text(text: str, config: PreprocessConfig | None = None) -> str:
    cfg = config or PreprocessConfig()
    value = unicodedata.normalize("NFKC", text)
    value = re.sub(r"\s+", " ", value).strip()
    if cfg.lowercase:
        value = value.lower()
    if cfg.expand_contractions:
        for pattern, repl in _CONTRACTION_RULES:
            value = pattern.sub(repl, value)
    tokens = _TOKEN_RE.findall(value)
    if cfg.lemmatize:
        tokens = _lemmatize_tokens(tokens)
    if cfg.remove_stopwords:
        stops = LIGHT_STOPWORDS if cfg.keep_negation else (LIGHT_STOPWORDS | NEGATION_WORDS)
        tokens = [tok for tok in tokens if tok not in stops]
    return " ".join(tokens)


class Course77Preprocessor(BaseEstimator, TransformerMixin):
    """Stateless sklearn-compatible text preprocessor for Course-77 pipelines."""

    def __init__(
        self,
        lowercase: bool = True,
        expand_contractions: bool = True,
        lemmatize: bool = True,
        remove_stopwords: bool = False,
        keep_negation: bool = True,
    ):
        self.lowercase = lowercase
        self.expand_contractions = expand_contractions
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.keep_negation = keep_negation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cfg = PreprocessConfig(
            lowercase=self.lowercase,
            expand_contractions=self.expand_contractions,
            lemmatize=self.lemmatize,
            remove_stopwords=self.remove_stopwords,
            keep_negation=self.keep_negation,
        )
        return [preprocess_text(str(item), cfg) for item in X]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw_banking77(raw_dir: str | Path = Path("data/raw/banking77")) -> tuple[list[dict], list[dict], list[str], dict]:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    cats_path = raw_dir / "categories.json"
    if not train_path.exists() or not test_path.exists() or not cats_path.exists():
        raise FileNotFoundError(f"missing raw banking77 files in {raw_dir}; run scripts/download_data.py --dataset banking77")
    categories = json.loads(cats_path.read_text(encoding="utf-8"))
    train_rows = list(csv.DictReader(train_path.open(encoding="utf-8")))
    test_rows = list(csv.DictReader(test_path.open(encoding="utf-8")))
    hashes = {
        "train.csv": _sha256_file(train_path),
        "test.csv": _sha256_file(test_path),
        "categories.json": _sha256_file(cats_path),
    }
    return train_rows, test_rows, categories, hashes


def build_course77_splits(
    raw_dir: str | Path = Path("data/raw/banking77"),
    processed_dir: str | Path = Path("data/processed/course77"),
    runs_dir: str | Path = Path("runs/course77"),
    *,
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Audit BANKING77 and construct deterministic stratified train_fit/dev splits."""
    deps = _require_classical()
    StratifiedShuffleSplit = deps["StratifiedShuffleSplit"]

    train_rows, test_rows, categories, source_hashes = load_raw_banking77(raw_dir)
    if len(categories) != 77:
        raise ValueError(f"expected 77 categories, got {len(categories)}")

    test_norm_map: dict[str, list[dict]] = defaultdict(list)
    test_blanks = 0
    for idx, row in enumerate(test_rows):
        if not row["text"].strip():
            test_blanks += 1
        test_norm_map[normalize_for_dedup(row["text"])].append(
            {"sample_id": f"test/{idx}", "label": row["category"], "text": row["text"]}
        )

    test_internal_dups = {k: v for k, v in test_norm_map.items() if len(v) > 1}

    seen_train_norms: dict[str, str] = {}
    train_internal_duplicates: list[dict] = []
    train_test_overlaps: list[dict] = []
    train_blanks: list[str] = []
    train_records: list[dict] = []

    for idx, row in enumerate(train_rows):
        sample_id = f"train/{idx}"
        text = row["text"]
        label = row["category"]
        if label not in categories:
            raise ValueError(f"unknown label {label!r} at {sample_id}")
        norm = normalize_for_dedup(text)
        eligible = True
        excluded_reason: str | None = None

        if not text.strip():
            eligible = False
            excluded_reason = "blank_text"
            train_blanks.append(sample_id)
        elif norm in test_norm_map:
            eligible = False
            excluded_reason = "train_test_normalized_overlap"
            train_test_overlaps.append({
                "train_sample_id": sample_id,
                "train_label": label,
                "normalized_text": norm,
                "test_matches": [
                    {"sample_id": m["sample_id"], "label": m["label"]} for m in test_norm_map[norm]
                ],
            })
        elif norm in seen_train_norms:
            eligible = False
            excluded_reason = "train_internal_duplicate"
            train_internal_duplicates.append({
                "duplicate_sample_id": sample_id,
                "kept_sample_id": seen_train_norms[norm],
                "label": label,
                "normalized_text": norm,
            })
        else:
            seen_train_norms[norm] = sample_id

        train_records.append({
            "sample_id": sample_id,
            "row_index": idx,
            "text": text,
            "label": label,
            "source_split": "train",
            "split": "excluded" if not eligible else "pending",
            "eligible": eligible,
            "excluded_reason": excluded_reason,
        })

    eligible_indices = [i for i, r in enumerate(train_records) if r["eligible"]]
    eligible_labels = [train_records[i]["label"] for i in eligible_indices]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=dev_ratio, random_state=seed)
    fit_pos, dev_pos = next(splitter.split(eligible_indices, eligible_labels))

    for pos in fit_pos:
        train_records[eligible_indices[pos]]["split"] = "train_fit"
    for pos in dev_pos:
        train_records[eligible_indices[pos]]["split"] = "dev"

    test_records = [
        {
            "sample_id": f"test/{idx}",
            "row_index": idx,
            "text": row["text"],
            "label": row["category"],
            "source_split": "test",
            "split": "test",
            "eligible": bool(row["text"].strip()),
            "excluded_reason": None if row["text"].strip() else "blank_text",
        }
        for idx, row in enumerate(test_rows)
    ]

    all_records = train_records + test_records

    # Verify strict disjointness across train_fit, dev, and official test.
    fit_norms = {normalize_for_dedup(r["text"]) for r in all_records if r["split"] == "train_fit"}
    dev_norms = {normalize_for_dedup(r["text"]) for r in all_records if r["split"] == "dev"}
    test_norms = {normalize_for_dedup(r["text"]) for r in all_records if r["split"] == "test"}
    assert not (fit_norms & dev_norms), "leakage between train_fit and dev"
    assert not (fit_norms & test_norms), "leakage between train_fit and official test"
    assert not (dev_norms & test_norms), "leakage between dev and official test"

    split_counts = dict(Counter(r["split"] for r in all_records))
    fit_label_counts = Counter(r["label"] for r in all_records if r["split"] == "train_fit")
    dev_label_counts = Counter(r["label"] for r in all_records if r["split"] == "dev")
    test_label_counts = Counter(r["label"] for r in all_records if r["split"] == "test")

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(
        processed_dir / "splits.json",
        {
            "model_scope": "course77",
            "seed": seed,
            "dev_ratio": dev_ratio,
            "source_sha256": source_hashes,
            "dedup_overlap_policy": (
                "Keep official test (3080) completely untouched. Exclude train rows matching "
                "official test after strip+casefold (6 rows) and redundant internal train "
                "duplicates (4 rows) from train_fit/dev before stratified splitting."
            ),
            "counts": split_counts,
            "samples": [{k: v for k, v in r.items() if k != "text"} for r in all_records],
        },
    )

    label_map = {
        "model_scope": "course77",
        "num_labels": len(categories),
        "id2label": {str(i): cat for i, cat in enumerate(categories)},
        "label2id": {cat: i for i, cat in enumerate(categories)},
    }
    (processed_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2) + "\n", encoding="utf-8"
    )

    audit_report = {
        "model_scope": "course77",
        "dataset": "banking77",
        "source_sha256": source_hashes,
        "seed": seed,
        "dev_ratio": dev_ratio,
        "raw_counts": {
            "train": len(train_rows),
            "test": len(test_rows),
            "labels": len(categories),
            "train_blanks": len(train_blanks),
            "test_blanks": test_blanks,
        },
        "duplicates_and_overlap": {
            "train_internal_duplicate_extra_rows": len(train_internal_duplicates),
            "train_internal_duplicates": train_internal_duplicates,
            "test_internal_duplicate_groups": len(test_internal_dups),
            "train_test_normalized_overlap_count": len(train_test_overlaps),
            "train_test_overlaps": train_test_overlaps,
            "policy": (
                "Official test is preserved unchanged. 6 train rows overlapping official test "
                "(after strip+casefold) and 4 extra internal duplicate train rows are marked "
                "eligible=False and excluded from train_fit/dev to prevent leakage."
            ),
        },
        "split_counts": split_counts,
        "disjoint_normalized_text_verified": True,
        "per_split_label_coverage": {
            "train_fit_labels": len(fit_label_counts),
            "train_fit_min_max": [min(fit_label_counts.values()), max(fit_label_counts.values())],
            "dev_labels": len(dev_label_counts),
            "dev_min_max": [min(dev_label_counts.values()), max(dev_label_counts.values())],
            "test_labels": len(test_label_counts),
            "test_min_max": [min(test_label_counts.values()), max(test_label_counts.values())],
        },
    }
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "course77_audit.json").write_text(
        json.dumps(audit_report, indent=2) + "\n", encoding="utf-8"
    )
    return all_records, audit_report


def load_course77_records_from_splits(
    raw_dir: str | Path = Path("data/raw/banking77"),
    processed_dir: str | Path = Path("data/processed/course77"),
) -> tuple[list[dict], dict]:
    """Load frozen Course-77 split IDs and join raw text deterministically."""
    splits_path = Path(processed_dir) / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"missing frozen splits manifest: {splits_path}")
    manifest = json.loads(splits_path.read_text(encoding="utf-8"))
    train_rows, test_rows, _, source_hashes = load_raw_banking77(raw_dir)
    if manifest.get("source_sha256") != source_hashes:
        raise ValueError("raw banking77 checksums do not match frozen splits.json")

    by_source = {"train": train_rows, "test": test_rows}
    records: list[dict] = []
    for sample in manifest["samples"]:
        raw_row = by_source[sample["source_split"]][sample["row_index"]]
        if raw_row["category"] != sample["label"]:
            raise ValueError(f"label mismatch at {sample['sample_id']}")
        records.append({**sample, "text": raw_row["text"]})
    return records, manifest


@dataclass(frozen=True)
class Course77RunSpec:
    experiment_id: str
    slug: str
    metrics_filename: str
    predictions_filename: str
    representation: str
    classifier_type: str
    preprocess: PreprocessConfig = PreprocessConfig()
    c_value: float = 1.0
    max_iter: int = 1000
    seed: int = 42


PACKAGE_B_SPECS: tuple[Course77RunSpec, ...] = (
    Course77RunSpec(
        experiment_id="E1",
        slug="e1_bow_lr",
        metrics_filename="e1_dev_metrics.json",
        predictions_filename="e1_dev_predictions.jsonl",
        representation="bow",
        classifier_type="logistic_regression",
        preprocess=PreprocessConfig(lemmatize=True),
        c_value=1.0,
        max_iter=1000,
    ),
    Course77RunSpec(
        experiment_id="E2",
        slug="e2_tfidf_lr",
        metrics_filename="e2_dev_metrics.json",
        predictions_filename="e2_dev_predictions.jsonl",
        representation="tfidf",
        classifier_type="logistic_regression",
        preprocess=PreprocessConfig(lemmatize=True),
        c_value=1.0,
        max_iter=1000,
    ),
    Course77RunSpec(
        experiment_id="E3",
        slug="e3_tfidf_svm",
        metrics_filename="e3_dev_metrics.json",
        predictions_filename="e3_dev_predictions.jsonl",
        representation="tfidf",
        classifier_type="linear_svm",
        preprocess=PreprocessConfig(lemmatize=True),
        c_value=1.0,
        max_iter=2000,
    ),
    Course77RunSpec(
        experiment_id="E2_no_lemma",
        slug="e2_tfidf_lr_no_lemma",
        metrics_filename="e2_no_lemma_dev_metrics.json",
        predictions_filename="e2_no_lemma_dev_predictions.jsonl",
        representation="tfidf",
        classifier_type="logistic_regression",
        preprocess=PreprocessConfig(lemmatize=False),
        c_value=1.0,
        max_iter=1000,
    ),
)


def _build_vectorizer_and_classifier(spec: Course77RunSpec, deps: dict):
    if spec.representation == "bow":
        vectorizer = deps["CountVectorizer"](ngram_range=(1, 1), min_df=1)
        rep_desc = "BoW (CountVectorizer unigram, min_df=1)"
    elif spec.representation == "tfidf":
        vectorizer = deps["TfidfVectorizer"](ngram_range=(1, 1), min_df=1)
        rep_desc = "TF-IDF (TfidfVectorizer unigram, min_df=1)"
    else:
        raise ValueError(f"unsupported representation: {spec.representation}")

    if spec.classifier_type == "logistic_regression":
        classifier = deps["LogisticRegression"](C=spec.c_value, max_iter=spec.max_iter, random_state=spec.seed)
        clf_desc = f"LogisticRegression(C={spec.c_value}, max_iter={spec.max_iter}, random_state={spec.seed})"
    elif spec.classifier_type == "linear_svm":
        classifier = deps["LinearSVC"](C=spec.c_value, max_iter=spec.max_iter, random_state=spec.seed)
        clf_desc = f"LinearSVC(C={spec.c_value}, max_iter={spec.max_iter}, random_state={spec.seed})"
    else:
        raise ValueError(f"unsupported classifier_type: {spec.classifier_type}")

    return vectorizer, rep_desc, classifier, clf_desc


def _prediction_scores(pipeline, texts: list[str]) -> tuple[np.ndarray, str]:
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(texts), dtype=float), "probability"
    return np.asarray(pipeline.decision_function(texts), dtype=float), "decision_margin"


def train_and_evaluate_experiment(
    records: list[dict],
    spec: Course77RunSpec,
    *,
    artifacts_dir: str | Path = Path("artifacts/course77"),
    runs_dir: str | Path = Path("runs/course77"),
    splits_path: str | Path = Path("data/processed/course77/splits.json"),
) -> dict:
    """Train one Course-77 pipeline strictly on train_fit and evaluate on dev."""
    deps = _require_classical()
    joblib = deps["joblib"]
    Pipeline = deps["Pipeline"]

    cfg = spec.preprocess
    fit_rows = [r for r in records if r["split"] == "train_fit"]
    dev_rows = [r for r in records if r["split"] == "dev"]
    if not fit_rows or not dev_rows:
        raise ValueError("train_fit and dev partitions must be non-empty")

    X_fit = [r["text"] for r in fit_rows]
    y_fit = [r["label"] for r in fit_rows]
    X_dev = [r["text"] for r in dev_rows]
    y_dev = [r["label"] for r in dev_rows]

    vectorizer, rep_desc, classifier, clf_desc = _build_vectorizer_and_classifier(spec, deps)
    pipeline = Pipeline([
        (
            "preprocessor",
            Course77Preprocessor(
                lowercase=cfg.lowercase,
                expand_contractions=cfg.expand_contractions,
                lemmatize=cfg.lemmatize,
                remove_stopwords=cfg.remove_stopwords,
                keep_negation=cfg.keep_negation,
            ),
        ),
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])

    t0 = perf_counter()
    pipeline.fit(X_fit, y_fit)
    train_seconds = perf_counter() - t0

    t1 = perf_counter()
    y_pred = pipeline.predict(X_dev)
    score_matrix, score_kind = _prediction_scores(pipeline, X_dev)
    eval_seconds = perf_counter() - t1

    classes = list(pipeline.named_steps["classifier"].classes_)
    class_to_idx = {label: i for i, label in enumerate(classes)}

    acc = float(deps["accuracy_score"](y_dev, y_pred))
    macro_p = float(deps["precision_score"](y_dev, y_pred, average="macro", zero_division=0))
    macro_r = float(deps["recall_score"](y_dev, y_pred, average="macro", zero_division=0))
    macro_f1 = float(deps["f1_score"](y_dev, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(deps["f1_score"](y_dev, y_pred, average="weighted", zero_division=0))

    per_class = deps["classification_report"](y_dev, y_pred, labels=classes, output_dict=True, zero_division=0)
    cm = deps["confusion_matrix"](y_dev, y_pred, labels=classes)

    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "gold": classes[i],
                    "pred": classes[j],
                    "count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda x: (-x["count"], x["gold"], x["pred"]))

    error_samples = []
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    preds_path = runs_dir / spec.predictions_filename
    with preds_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, pred_label, score_vec in zip(dev_rows, y_pred, score_matrix):
            pred_score = float(score_vec[class_to_idx[pred_label]])
            gold_score = float(score_vec[class_to_idx[row["label"]]])
            correct = bool(pred_label == row["label"])
            rec = {
                "sample_id": row["sample_id"],
                "gold": row["label"],
                "prediction": str(pred_label),
                "score_kind": score_kind,
                "confidence": round(pred_score, 6),
                "gold_prob": round(gold_score, 6),
                "correct": correct,
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not correct:
                error_samples.append({
                    **rec,
                    "text": row["text"],
                    "preprocessed_text": preprocess_text(row["text"], cfg),
                })

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"{spec.slug}.joblib"
    joblib.dump(pipeline, model_path)

    # Verify reload from disk and collect 3 smoke examples from dev.
    reloaded = joblib.load(model_path)
    sample_inputs = [dev_rows[0], dev_rows[len(dev_rows) // 2], dev_rows[-1]]
    smoke_examples = []
    for row in sample_inputs:
        pred = str(reloaded.predict([row["text"]])[0])
        smoke_scores, _ = _prediction_scores(reloaded, [row["text"]])
        top_score = float(np.max(smoke_scores[0]))
        smoke_examples.append({
            "sample_id": row["sample_id"],
            "text": row["text"],
            "gold": row["label"],
            "predicted": pred,
            "score_kind": score_kind,
            "confidence": round(top_score, 6),
        })

    vocab_size = len(pipeline.named_steps["vectorizer"].vocabulary_)
    splits_sha256 = _sha256_file(Path(splits_path)) if Path(splits_path).exists() else None
    model_sha256 = _sha256_file(model_path)

    config_payload = {
        "experiment_id": spec.experiment_id,
        "model_scope": "course77",
        "representation": rep_desc,
        "classifier": clf_desc,
        "fit_split": "train_fit",
        "eval_split": "dev",
        "official_test_touched": False,
        "preprocessing": asdict(cfg),
        "preprocessing_motivation": {
            "normalization": "Unicode NFKC, whitespace collapse, lowercasing, and contraction expansion (preserving negation) reduce surface sparsity.",
            "tokenization": "Alphanumeric token extraction cleanly separates punctuation while retaining banking terms (pin, atm, 2fa).",
            "lemmatization": "WordNet verb+noun lemmatization collapses inflectional variants (cards->card, charged->charge); toggled off in ablation to measure exact impact.",
        },
        "vocab_size": vocab_size,
        "train_fit_samples": len(fit_rows),
        "dev_samples": len(dev_rows),
        "splits_sha256": splits_sha256,
        "artifact_path": model_path.as_posix(),
        "artifact_sha256": model_sha256,
        "smoke_examples": smoke_examples,
    }
    (artifacts_dir / f"{spec.slug}_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics_path = runs_dir / spec.metrics_filename
    metrics_payload = {
        "experiment_id": spec.experiment_id,
        "model_scope": "course77",
        "representation": rep_desc,
        "classifier": clf_desc,
        "preprocessing": asdict(cfg),
        "eval_split": "dev",
        "official_test_touched": False,
        "fit_samples": len(fit_rows),
        "eval_samples": len(dev_rows),
        "vocab_size": vocab_size,
        "train_seconds": round(train_seconds, 4),
        "eval_seconds": round(eval_seconds, 4),
        "metrics": {
            "accuracy": round(acc, 6),
            "macro_precision": round(macro_p, 6),
            "macro_recall": round(macro_r, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
        },
        "error_count": len(error_samples),
        "top_confused_pairs": confused_pairs[:15],
        "error_samples_first_15": error_samples[:15],
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist(),
        },
        "per_class_report": per_class,
        "smoke_examples": smoke_examples,
        "artifact_path": model_path.as_posix(),
        "metrics_path": metrics_path.as_posix(),
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics_payload


def train_and_evaluate_e1(
    records: list[dict],
    *,
    artifacts_dir: str | Path = Path("artifacts/course77"),
    runs_dir: str | Path = Path("runs/course77"),
    splits_path: str | Path = Path("data/processed/course77/splits.json"),
    preprocess_config: PreprocessConfig | None = None,
    c_value: float = 1.0,
    max_iter: int = 1000,
    seed: int = 42,
) -> dict:
    """Backward-compatible wrapper for Package A E1 (BoW + LogisticRegression)."""
    spec = Course77RunSpec(
        experiment_id="E1",
        slug="e1_bow_lr",
        metrics_filename="e1_dev_metrics.json",
        predictions_filename="e1_dev_predictions.jsonl",
        representation="bow",
        classifier_type="logistic_regression",
        preprocess=preprocess_config or PreprocessConfig(),
        c_value=c_value,
        max_iter=max_iter,
        seed=seed,
    )
    return train_and_evaluate_experiment(
        records,
        spec,
        artifacts_dir=artifacts_dir,
        runs_dir=runs_dir,
        splits_path=splits_path,
    )


def run_package_b_suite(
    records: list[dict],
    *,
    artifacts_dir: str | Path = Path("artifacts/course77"),
    runs_dir: str | Path = Path("runs/course77"),
    splits_path: str | Path = Path("data/processed/course77/splits.json"),
    specs: tuple[Course77RunSpec, ...] = PACKAGE_B_SPECS,
) -> dict:
    """Run E1, E2, E3, and controlled lemmatization ablation on the exact same split."""
    results_by_id: dict[str, dict] = {}
    table_rows: list[dict] = []

    for spec in specs:
        res = train_and_evaluate_experiment(
            records,
            spec,
            artifacts_dir=artifacts_dir,
            runs_dir=runs_dir,
            splits_path=splits_path,
        )
        results_by_id[spec.experiment_id] = res
        table_rows.append({
            "experiment_id": spec.experiment_id,
            "representation": res["representation"],
            "classifier": res["classifier"],
            "lemmatize": spec.preprocess.lemmatize,
            "vocab_size": res["vocab_size"],
            "accuracy": res["metrics"]["accuracy"],
            "macro_precision": res["metrics"]["macro_precision"],
            "macro_recall": res["metrics"]["macro_recall"],
            "macro_f1": res["metrics"]["macro_f1"],
            "weighted_f1": res["metrics"]["weighted_f1"],
            "error_count": res["error_count"],
            "train_seconds": res["train_seconds"],
            "eval_seconds": res["eval_seconds"],
            "artifact_path": res["artifact_path"],
            "metrics_path": res["metrics_path"],
        })

    def _delta(later: str, base: str, changed: str) -> dict | None:
        if later not in results_by_id or base not in results_by_id:
            return None
        a, b = results_by_id[later]["metrics"], results_by_id[base]["metrics"]
        return {
            "base": base,
            "variant": later,
            "changed_factor": changed,
            "delta_accuracy": round(a["accuracy"] - b["accuracy"], 6),
            "delta_macro_f1": round(a["macro_f1"] - b["macro_f1"], 6),
        }

    splits_sha256 = _sha256_file(Path(splits_path)) if Path(splits_path).exists() else None
    summary = {
        "package": "B",
        "model_scope": "course77",
        "eval_split": "dev",
        "official_test_touched": False,
        "splits_sha256": splits_sha256,
        "reproduce_command": ".\\.venv\\Scripts\\python.exe scripts/run_course77.py --suite package_b",
        "comparison_table": table_rows,
        "controlled_comparisons": {
            "representation_tfidf_vs_bow": _delta("E2", "E1", "CountVectorizer -> TfidfVectorizer (LogisticRegression fixed)"),
            "classifier_svm_vs_lr": _delta("E3", "E2", "LogisticRegression -> LinearSVC (TfidfVectorizer fixed)"),
            "ablation_lemmatize_off_vs_on_e2": _delta("E2_no_lemma", "E2", "lemmatize=True -> lemmatize=False (E2 TF-IDF+LR fixed)"),
        },
    }
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "package_b_comparison_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
