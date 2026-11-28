"""Fit R7 on one turn per public-train dialogue. No validation examples or responses enter training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.passage_ranker import PassageRanker, SCHEMA
from supportlens.query_memory import training_bundle
from supportlens.retrieval import BM25Chunks


def training_rows(queries, qrels, splits):
    train = set(splits["train"])
    if train.intersection(splits["dev"]) or train.intersection(splits["heldout"]):
        raise ValueError("overlapping train/evaluation dialogues")
    gold = {r["query_id"]: set(r["strict"]) for r in qrels}
    ids = [q["query_id"] for q in queries]
    if len(gold) != len(qrels) or len(set(ids)) != len(ids) or set(ids) != gold.keys():
        raise ValueError("query and qrels IDs differ")
    seen = set()
    for q in sorted(queries, key=lambda q: hashlib.sha256(q["query_id"].encode()).digest()):
        key, dialogue = q["query_id"], q["dialogue_id"]
        if dialogue not in train or key.rsplit(":", 1)[0] != dialogue:
            raise ValueError("query from outside a training dialogue")
        if dialogue not in seen and q["valid_input"] and gold[key]:
            seen.add(dialogue)
            yield q, gold[key]


def export(model, base):
    trees = []
    for est in model.estimators_.ravel():
        t = est.tree_
        if t.max_depth > 3:
            raise ValueError("unsupported tree depth")
        trees.append({k: v.tolist() for k, v in
                      [("left", t.children_left), ("right", t.children_right), ("feature", t.feature),
                       ("threshold", t.threshold), ("value", t.value[:, 0, 0])]})
    return {"schema": SCHEMA, "base": base.version, "features": PassageRanker.feature_names(),
            "learning_rate": model.learning_rate, "trees": trees}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bundle = args.data / "processed/docs_qa"
    base = BM25Chunks.from_bundle(bundle, args.data / "tokenizers/minilm/tokenizer.json")
    ranker = PassageRanker(base)
    features, labels, used, missed, weights = [], [], [], [], []
    for q, gold in training_rows(*training_bundle(base, bundle)):
        ids, x = ranker.features(q["query_text"])
        positive = np.isin(ranker.keys[ids], list(gold))
        if not positive.any() or positive.all():
            missed.append(q["query_id"])
            continue
        features.append(x)
        labels.append(positive)
        weights.append(np.where(positive, 0.5/positive.sum(), 0.5/(~positive).sum()))
        used.append(q["query_id"])
    x, y = np.vstack(features), np.concatenate(labels)
    model = GradientBoostingClassifier(n_estimators=48, max_depth=3, learning_rate=0.1, random_state=42,
                                       min_samples_leaf=20, subsample=0.6, init="zero")
    model.fit(x, y, sample_weight=np.concatenate(weights))
    artifact = export(model, base)
    portable = PassageRanker(base, artifact)
    np.testing.assert_allclose(portable.decision_function(x[::97]), model.decision_function(x[::97]), atol=1e-9)
    (args.out/"model.json").write_text(json.dumps(artifact, separators=(",", ":"))+"\n", encoding="utf-8", newline="\n")
    summary = {"fit_dialogues": len(used), "excluded_without_both_classes": len(missed),
               "pairs": len(y), "positives": int(y.sum()), "version": portable.version,
               "sklearn": sklearn.__version__, "numpy": np.__version__,
               "bundle_sha256": hashlib.sha256((bundle/"manifest.json").read_bytes()).hexdigest()}
    (args.out/"training_metrics.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
