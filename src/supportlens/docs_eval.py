"""Dev-only retrieval experiments; dialogue-cluster bootstrap preserves dependence."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import examples


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if len(rendered) > 100_000:
        # Full curves are machine-readable artifacts; avoid multi-megabyte indentation.
        rendered = json.dumps(report, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    path.write_text(rendered + "\n",
                    encoding="utf-8", newline="\n")


def load_dev(bundle, *, raw_validation=None):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    def verified(name):
        value = (bundle / name).read_bytes()
        if hashlib.sha256(value).hexdigest() != manifest["files_sha256"][name]:
            raise ValueError(f"bundle bytes changed: {name}")
        return value.decode("utf-8")
    split = json.loads(verified("splits.json"))
    dev_ids = set(split["dialogues"]["dev"])
    if dev_ids & (set(split["dialogues"]["train"]) | set(split["dialogues"]["heldout"])):
        raise ValueError("overlapping dialogue splits")
    queries = [json.loads(line) for line in verified("queries.dev.jsonl").splitlines()]
    qrels = {r["query_id"]: r for r in map(json.loads, verified("qrels.dev.jsonl").splitlines())}
    if len(queries) != len(qrels) or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("query/qrels IDs must be unique and aligned")
    if any(q["dialogue_id"] not in dev_ids or q["query_id"] not in qrels for q in queries):
        raise ValueError("require dev queries only")
    if raw_validation is not None:
        raw_validation = Path(raw_validation)
        if hashlib.sha256(raw_validation.read_bytes()).hexdigest() != manifest["sources_sha256"]["validation"]:
            raise ValueError("raw validation source changed")
        raw = {r["query_id"]: r for r in examples(raw_validation, dialogue_ids=dev_ids)}
        if set(raw) != {q["query_id"] for q in queries}:
            raise ValueError("raw dev queries do not match the bundle")
        for q in queries:
            r = raw[q["query_id"]]
            if r["question"].strip() != q["question"]:
                raise ValueError("raw question differs")
            # No target response or references enter retrieval records.
            q["history"] = r["history"]
    return manifest, queries, qrels


def query_metrics(ids, gold):
    gold = set(gold)
    if not gold:
        return None
    return {**{f"hit_at_{k}": float(bool(gold & set(ids[:k]))) for k in (1, 3, 5)},
            "gold_recall_at_5": len(gold & set(ids[:5])) / len(gold),
            "mrr_at_10": next((1 / i for i, key in enumerate(ids[:10], 1) if key in gold), 0.0)}


def aggregate_metrics(rows):
    defined = [r for r in rows if r is not None]
    return {"evaluated_queries_with_positive_gold": len(defined),
            "queries_without_positive_gold": len(rows) - len(defined),
            **{k: float(np.mean([r[k] for r in defined])) for k in defined[0]}} if defined else {
                "evaluated_queries_with_positive_gold": 0, "queries_without_positive_gold": len(rows)}


def paired_dialogue_bootstrap(dialogues, left, right, *, repeats=2000, seed=42):
    if not (len(dialogues) == len(left) == len(right)) or not dialogues or repeats < 1:
        raise ValueError("nonempty aligned paired values and positive repeats required")
    sums, counts = defaultdict(float), defaultdict(int)
    for d, a, b in zip(dialogues, left, right):
        sums[d] += a - b
        counts[d] += 1
    keys = sorted(sums)
    s, n = np.array([sums[d] for d in keys]), np.array([counts[d] for d in keys])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        indices = rng.integers(0, len(keys), len(keys))
        boot.append(float(s[indices].sum() / n[indices].sum()))
    return {"delta_left_minus_right": float(s.sum() / n.sum()),
            "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "unit": "dialogue", "estimator": "query-weighted mean after cluster resampling",
            "dialogues": len(keys), "repeats": repeats, "seed": seed}


def binary_auroc(scores, successes):
    """Mann-Whitney AUROC with average ranks for ties, no sklearn dependency."""
    scores, successes = np.asarray(scores, dtype=float), np.asarray(successes, dtype=bool)
    if scores.ndim != 1 or scores.shape != successes.shape or not np.isfinite(scores).all():
        raise ValueError("finite aligned one-dimensional scores required")
    pos = int(successes.sum())
    if pos == 0 or pos == len(scores):
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[successes].sum() - pos * (pos + 1) / 2) / (pos * (len(scores) - pos)))


def visible_gold_hit(prompt, chunks_by_id, domain, references, raw_docs, *, minimum=0.5):
    """Check raw-span support in the exact visible evidence prefix, excluding title.

    Like strict qrels this is >=50% of any one gold span, not all-claims support.
    It audits prompt packing without calling or judging an LLM.
    """
    if not 0 < minimum <= 1:
        raise ValueError("minimum span coverage outside (0,1]")
    for key, count in prompt.visible_characters.items():
        chunk = chunks_by_id[key]
        if not 0 <= count <= len(chunk.text):
            raise ValueError("invalid visible character count")
        prefix_length = len(chunk.text) - (chunk.end - chunk.start)
        visible_end = chunk.start + max(0, count - prefix_length)
        if chunk.domain != domain:
            continue
        for reference in references:
            if reference["doc_id"] != chunk.doc_id:
                continue
            doc = raw_docs[domain][chunk.doc_id]
            span = doc["spans"][str(reference["id_sp"])]
            start, end = span["start_sp"], span["end_sp"]
            if not 0 <= start < end <= len(doc["doc_text"]):
                raise ValueError("invalid gold span")
            overlap = max(0, min(end, visible_end) - max(start, chunk.start))
            if overlap / (end - start) >= minimum:
                return True
    return False
