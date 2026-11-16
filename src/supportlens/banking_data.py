"""Build aligned frozen vectors from the public Handoff-50 source partitions."""

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .banking import BankingEncoder


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
                valid_input = bool(text.strip())
                rows.append({"sample_id": f"{group}/{source_split}/{i}", "text": text,
                             "label": label, "original_intent": original,
                             "split": split, "group": group,
                             "valid_input": valid_input,
                             "fit_eligible": split == "train" and group == "id" and valid_input})
    train_labels = {r["label"] for r in rows if r["fit_eligible"]}
    if len(train_labels) != 50:
        raise ValueError(f"expected the public 50-label ID train set, got {len(train_labels)}")
    if any(r["label"] not in train_labels for r in rows if r["group"] == "id"):
        raise ValueError("ID dev/test label absent from the 50-label train set")
    return rows, sources


def build_cache(rows: list[dict], encoder: BankingEncoder, destination: str | Path,
                *, source_hashes: dict[str, str], batch_size: int = 32) -> dict:
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("non-empty uniquely identified rows required")
    rows = [dict(r, valid_input=bool(r["text"].strip())) for r in rows]
    if any(r["fit_eligible"] and not r["valid_input"] for r in rows):
        raise ValueError("blank source rows cannot be used to fit a classifier")
    started = perf_counter()
    valid = np.asarray([r["valid_input"] for r in rows])
    vectors = np.zeros((len(rows), encoder.config.dimension), dtype=np.float32)
    vectors[valid] = encoder.encode([r["text"] for r in rows if r["valid_input"]], batch_size=batch_size)
    elapsed = perf_counter() - started
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / "embeddings.npz"
    arrays = {key: np.asarray([r[key] for r in rows]) for key in (
        "sample_id", "label", "original_intent", "split", "group", "fit_eligible", "valid_input")}
    np.savez_compressed(cache, embeddings=vectors, **arrays)
    manifest = {"model_scope": "handoff50", "encoder": asdict(encoder.config),
                "encoder_fingerprint": encoder.config.fingerprint,
                "source_sha256": source_hashes, "rows": len(rows), "shape": list(vectors.shape),
                "counts": dict(Counter(r["split"] + "/" + r["group"] for r in rows)),
                "encoded_rows": int(valid.sum()),
                "invalid_input_ids": [r["sample_id"] for r in rows if not r["valid_input"]],
                "invalid_input_policy": "zero sentinel, not an embedding; never fit/score/reference-bank; "
                                        "retain IDs and count forced handoff separately in evaluation",
                "encode_seconds": elapsed, "device": encoder.device,
                "npz_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                "fit_rule": "split=train AND group=id only; dev selects; test reports"}
    (destination / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    splits = {"model_scope": "handoff50", "source_sha256": source_hashes,
              "samples": [{k: v for k, v in r.items() if k != "text"} for r in rows]}
    (destination / "splits.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return manifest
