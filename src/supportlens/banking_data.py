"""Build aligned frozen vectors and C-01/C-02 masks from Handoff-50 source partitions."""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .banking import BankingEncoder


EXCLUDED_TRAIN_LEAKS = frozenset({
    "id/train/3997",  # Exact & MiniLM token dup of id/valid/1017 (change_pin)
    "id/train/4018",  # Exact & MiniLM token dup of id/test/1353 (change_pin)
    "id/train/2128",  # MiniLM token dup & label conflict with id/test/721
    "id/train/5823",  # MiniLM token dup of id/test/1994 (country_support)
})

EXCLUDED_DEV_DUPLICATES = frozenset({
    "id/valid/1091",     # MiniLM token dup of id/test/1415 (transfer_fee_charged)
    "id_oos/valid/211",  # Exact & MiniLM token dup of id_oos/test/419
    "id_oos/valid/288",  # Exact & MiniLM token dup of id_oos/test/595
    "id_oos/valid/310",  # Exact & MiniLM token dup of id_oos/test/638
})


def write_split_manifest(path: str | Path, payload: dict) -> None:
    """Valid JSON, one compact sample per line so mask changes stay local in Git."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {k: v for k, v in payload.items() if k != "samples"}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, ensure_ascii=False, separators=(",", ":"))[:-1])
        handle.write(("," if header else "") + '"samples":[\n')
        for i, row in enumerate(payload["samples"]):
            handle.write((",\n" if i else "") + json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n]}\n")


def load_handoff_rows(root: str | Path) -> tuple[list[dict], dict[str, str]]:
    root = Path(root)
    rows, sources = [], {}
    for split, source_split in (("train", "train"), ("dev", "valid"), ("test", "test")):
        for group, subdir in (("id", ""), ("id_oos", "id-oos"), ("ood", "ood-oos")):
            if split == "train" and group != "id":
                continue  # Never fit/reference-bank with the 27 withheld intents or OOD.
            folder = root / subdir / source_split
            files = [folder / "seq.in", folder / "label"]
            texts, labels = [p.read_text(encoding="utf-8").splitlines() for p in files]
            if len(texts) != len(labels) or not texts:
                raise ValueError(f"unaligned/empty source: {folder}")
            originals = labels
            original_path = folder / "label_original"
            if original_path.exists():
                files.append(original_path)
                originals = original_path.read_text(encoding="utf-8").splitlines()
            if len(originals) != len(texts):
                raise ValueError(f"unaligned original labels: {folder}")
            for p in files:
                sources[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            for i, (text, label, original) in enumerate(zip(texts, labels, originals)):
                sample_id = f"{group}/{source_split}/{i}"
                valid_input = bool(text.strip())
                is_train_leak = sample_id in EXCLUDED_TRAIN_LEAKS
                is_dev_dup = sample_id in EXCLUDED_DEV_DUPLICATES
                rows.append({
                    "sample_id": sample_id,
                    "text": text,
                    "label": label,
                    "original_intent": original,
                    "split": split,
                    "group": group,
                    "valid_input": valid_input,
                    "calibration": False,
                    "fit_eligible": split == "train" and group == "id" and valid_input and not is_train_leak,
                    "dev_eligible": split == "dev" and valid_input and not is_dev_dup,
                })
    train_labels = {r["label"] for r in rows if r["fit_eligible"]}
    if len(train_labels) != 50:
        raise ValueError(f"expected the public 50-label ID train set, got {len(train_labels)}")
    if any(r["label"] not in train_labels for r in rows if r["group"] == "id"):
        raise ValueError("ID dev/test label absent from the 50-label train set")
    return rows, sources


def _group_key_fn(texts: list[str], tokenizer_path: Path | None, max_seq_length: int) -> list[tuple[int, ...] | str]:
    if tokenizer_path is not None:
        tok_path = Path(tokenizer_path)
        if not tok_path.exists():
            raise FileNotFoundError(
                f"Missing pinned MiniLM tokenizer at {tok_path}. Run: python scripts/fetch_tokenizers.py"
            )
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_file(str(tok_path))
        tok.enable_truncation(max_length=max_seq_length)
        tok.no_padding()
        encodings = tok.encode_batch([t.strip() for t in texts], add_special_tokens=True)
        return [tuple(enc.ids) for enc in encodings]
    return [" ".join(t.strip().casefold().split()) for t in texts]


def assign_calibration_split(
    rows: list[dict],
    *,
    fraction: float = 0.10,
    seed: int = 42,
    tokenizer_path: str | Path | None = Path("data/tokenizers/minilm/tokenizer.json"),
    max_seq_length: int = 256,
) -> dict:
    """Split ~fraction of valid fit_eligible ID-train rows into calibration (group-stratified)."""
    if not (0.0 < fraction < 0.5):
        raise ValueError("calibration fraction must be in (0, 0.5)")

    for r in rows:
        r["calibration"] = False

    eligible_indices = [i for i, r in enumerate(rows) if r["fit_eligible"]]
    if not eligible_indices:
        raise ValueError("no fit_eligible rows available for calibration split")

    keys = _group_key_fn(
        [rows[i]["text"] for i in eligible_indices],
        Path(tokenizer_path) if tokenizer_path is not None else None,
        max_seq_length,
    )

    # Group eligible indices by (label, token_id_group_key) so duplicates stay together.
    # Also detect if the same token_id_key appears across different labels (would be a conflict).
    key_to_indices: dict[tuple[int, ...] | str, list[int]] = defaultdict(list)
    for idx, key in zip(eligible_indices, keys):
        key_to_indices[key].append(idx)

    label_groups: dict[str, list[list[int]]] = defaultdict(list)
    for key, idxs in key_to_indices.items():
        labels_in_group = {rows[i]["label"] for i in idxs}
        if len(labels_in_group) > 1:
            raise ValueError(f"conflicting labels in identical token group: {[rows[i]['sample_id'] for i in idxs]}")
        label_groups[next(iter(labels_in_group))].append(sorted(idxs))

    rng = np.random.default_rng(seed)
    cal_indices: set[int] = set()
    per_label: dict[str, dict[str, int]] = {}
    grouped_multi_sample_clusters = []

    for label in sorted(label_groups):
        groups = sorted(label_groups[label], key=lambda g: rows[g[0]]["sample_id"])
        total_label = sum(len(g) for g in groups)
        for g in groups:
            if len(g) > 1:
                grouped_multi_sample_clusters.append({
                    "label": label,
                    "sample_ids": [rows[i]["sample_id"] for i in g],
                })

        if total_label < 2 or len(groups) < 2:
            per_label[label] = {"total": total_label, "fit": total_label, "calibration": 0}
            continue

        target = max(1, round(total_label * fraction))
        perm = rng.permutation(len(groups))
        chosen = 0
        for g_idx in perm:
            if chosen >= target:
                break
            grp = groups[int(g_idx)]
            if total_label - (chosen + len(grp)) < 1:
                continue  # Must keep at least 1 sample in fit for this label.
            cal_indices.update(grp)
            chosen += len(grp)
        per_label[label] = {"total": total_label, "fit": total_label - chosen, "calibration": chosen}

    for i in cal_indices:
        rows[i]["calibration"] = True
        rows[i]["fit_eligible"] = False

    # Verify invariants
    fit_rows = [r for r in rows if r["fit_eligible"]]
    cal_rows = [r for r in rows if r["calibration"]]
    if any(r["fit_eligible"] and r["calibration"] for r in rows):
        raise AssertionError("fit_eligible and calibration must be disjoint")
    if any(r["split"] != "train" or r["group"] != "id" or not r["valid_input"] for r in cal_rows):
        raise AssertionError("calibration must contain only valid ID train rows")
    if len({r["label"] for r in fit_rows}) != len(label_groups):
        raise AssertionError("fit split lost a label during calibration partitioning")

    # Verify no token group is split across fit and calibration
    fit_keys = {k for idx, k in zip(eligible_indices, keys) if idx not in cal_indices}
    cal_keys = {k for idx, k in zip(eligible_indices, keys) if idx in cal_indices}
    if fit_keys & cal_keys:
        raise AssertionError("duplicate token-ID group split between fit and calibration")

    for cluster in grouped_multi_sample_clusters:
        first_id = cluster["sample_ids"][0]
        first_row = next(r for r in rows if r["sample_id"] == first_id)
        cluster["assigned_partition"] = "calibration" if first_row["calibration"] else "fit"

    return {
        "seed": seed,
        "target_fraction": fraction,
        "actual_fraction": len(cal_rows) / len(eligible_indices),
        "eligible_before_calibration": len(eligible_indices),
        "fit_rows": len(fit_rows),
        "calibration_rows": len(cal_rows),
        "labels_in_fit": len({r["label"] for r in fit_rows}),
        "labels_in_calibration": len({r["label"] for r in cal_rows}),
        "grouped_multi_sample_clusters": grouped_multi_sample_clusters,
        "per_label_counts": per_label,
    }


def build_cache(rows: list[dict], encoder: BankingEncoder, destination: str | Path,
                *, source_hashes: dict[str, str], batch_size: int = 32,
                calibration_policy: dict | None = None) -> dict:
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("non-empty uniquely identified rows required")
    rows = [
        dict(
            r,
            valid_input=bool(r["text"].strip()),
            calibration=bool(r.get("calibration", False)),
            dev_eligible=bool(r.get("dev_eligible", r["split"] == "dev" and bool(r["text"].strip()))),
        )
        for r in rows
    ]
    if any(r["fit_eligible"] and not r["valid_input"] for r in rows):
        raise ValueError("blank source rows cannot be used to fit a classifier")
    if any(r["fit_eligible"] and r["calibration"] for r in rows):
        raise ValueError("calibration rows cannot also be marked fit_eligible")
    started = perf_counter()
    valid = np.asarray([r["valid_input"] for r in rows])
    vectors = np.zeros((len(rows), encoder.config.dimension), dtype=np.float32)
    vectors[valid] = encoder.encode([r["text"] for r in rows if r["valid_input"]], batch_size=batch_size)
    elapsed = perf_counter() - started
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "embeddings.npz"
    arrays = {
        key: np.asarray([r[key] for r in rows])
        for key in (
            "sample_id", "label", "original_intent", "split", "group",
            "fit_eligible", "calibration", "valid_input", "dev_eligible",
        )
    }
    np.savez_compressed(cache, embeddings=vectors, **arrays)
    manifest = {
        "model_scope": "handoff50",
        "encoder": asdict(encoder.config),
        "encoder_fingerprint": encoder.config.fingerprint,
        "source_sha256": source_hashes,
        "rows": len(rows),
        "shape": list(vectors.shape),
        "counts": dict(Counter(r["split"] + "/" + r["group"] for r in rows)),
        "fit_rows": int(sum(r["fit_eligible"] for r in rows)),
        "calibration_rows": int(sum(r["calibration"] for r in rows)),
        "encoded_rows": int(valid.sum()),
        "invalid_input_ids": [r["sample_id"] for r in rows if not r["valid_input"]],
        "excluded_train_leak_ids": sorted(EXCLUDED_TRAIN_LEAKS),
        "excluded_dev_duplicate_ids": sorted(EXCLUDED_DEV_DUPLICATES),
        "invalid_input_policy": (
            "zero sentinel, not an embedding; never fit/score/reference-bank; "
            "retain IDs and count forced handoff separately in evaluation"
        ),
        "encode_seconds": elapsed,
        "device": encoder.device,
        "timing_scope": "encode call includes first model load and any required download; excludes environment setup",
        "npz_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "fit_rule": (
            "fit_eligible = (split=train AND group=id AND valid_input AND not C-01 leak AND not C-02 calibration); "
            "calibration = 10% group-stratified ID train (seed=42); dev selects (excluding C-01 dev dups); test reports"
        ),
    }
    (destination / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    splits = {
        "model_scope": "handoff50",
        **({"calibration_policy": calibration_policy} if calibration_policy is not None else {}),
        "source_sha256": source_hashes,
        "samples": [{k: v for k, v in r.items() if k != "text"} for r in rows],
    }
    write_split_manifest(destination / "splits.json", splits)
    return manifest
