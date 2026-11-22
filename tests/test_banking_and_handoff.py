from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from fastapi.testclient import TestClient
import numpy as np
import pytest

from supportlens.api import create_app
from supportlens.banking import BankingEncoder, BankingPipeline, BankingPrediction, EncoderConfig
from supportlens.banking_data import build_cache, load_handoff_rows
from supportlens.smoke import run_package


class OfflineEncoder:
    """Deterministic test double, never passed off as pretrained embeddings."""
    def encode(self, texts, **kwargs):
        result = np.asarray([[1, 1 + len(t), 1 + sum(t.encode()), 2] for t in texts], dtype=np.float32)
        if kwargs["normalize_embeddings"]:
            result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result


@pytest.fixture
def encoder():
    config = EncoderConfig("offline-test-double", "a" * 40, dimension=4)
    return BankingEncoder(config, backend=OfflineEncoder())


def test_cache_and_live_predictions_share_identical_preprocessing_and_vectors(tmp_path, encoder):
    rows = [{"sample_id": "id/train/0", "text": "  card replacement  ", "label": "card",
             "original_intent": "card", "split": "train", "group": "id", "fit_eligible": True}]
    manifest = build_cache(rows, encoder, tmp_path, source_hashes={})
    with np.load(tmp_path / "embeddings.npz", allow_pickle=False) as cache:
        expected = cache["embeddings"][0]
        assert cache["sample_id"].tolist() == ["id/train/0"]
    seen = []
    def head(vector, *, method):
        seen.append(vector)
        return BankingPrediction("card", 0.8, "accepted", "dev_gate_passed", "fake-test-head")
    pipeline = BankingPipeline(encoder, head, expected_encoder_fingerprint=manifest["encoder_fingerprint"])
    with TestClient(create_app(banking=pipeline)) as client:
        result = client.post("/v1/assist", json={
            "mode": "banking_triage", "question": rows[0]["text"]})
    assert result.status_code == 200 and result.json()["predicted_intent"] == "card"
    assert len(seen) == 2  # startup warm-up, then the user's request
    np.testing.assert_allclose(seen[-1], expected, atol=1e-6)
    assert manifest["npz_sha256"] == hashlib.sha256((tmp_path / "embeddings.npz").read_bytes()).hexdigest()


def test_wrong_encoder_is_rejected_before_serving(encoder):
    with pytest.raises(ValueError, match="do not match"):
        BankingPipeline(encoder, lambda v: None, expected_encoder_fingerprint="wrong")
    with pytest.raises(ValueError, match="non-blank"):
        encoder.encode([" "])
    assert encoder.encode([]).shape == (0, 4)


def test_source_partitions_never_turn_withheld_train_intents_into_fit_rows(tmp_path):
    def source(folder, texts, labels):
        folder.mkdir(parents=True)
        (folder / "seq.in").write_text("\n".join(texts) + "\n", encoding="utf-8")
        (folder / "label").write_text("\n".join(labels) + "\n", encoding="utf-8")
    for split in ["train", "valid", "test"]:
        source(tmp_path / split, [f"{split} question {i}" for i in range(50)], [f"intent{i}" for i in range(50)])
        for group in ["id-oos", "ood-oos"]:
            source(tmp_path / group / split, ["withheld input"], ["oos"])
    rows, hashes = load_handoff_rows(tmp_path)
    assert len(rows) == 154
    assert sum(r["fit_eligible"] for r in rows) == 50
    assert not any(r["split"] == "train" and r["group"] != "id" for r in rows)
    assert not any(name.startswith(("id-oos/train/", "ood-oos/train/")) for name in hashes)
    assert len({r["sample_id"] for r in rows}) == len(rows)


def test_smoke_rejects_an_artifact_whose_predictions_changed(tmp_path, monkeypatch):
    (tmp_path / "test_head.py").write_text(
        'def predict(sample, *, package_dir):\n return {"intent": "card", "score": 0.8}\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({"smoke_entrypoint": "test_head:predict"}))
    samples = tmp_path / "smoke_samples.jsonl"
    samples.write_text(json.dumps({"input": {"question": "card?"}, "expected": {"intent": "card"}}) + "\n")
    assert run_package(tmp_path)["status"] == "pass"
    samples.write_text(json.dumps({"input": {"question": "card?"}, "expected": {"intent": "transfer"}}) + "\n")
    with pytest.raises(ValueError, match="value/type differs"):
        run_package(tmp_path)


def test_git_can_track_ids_small_models_and_metrics_but_not_weights_or_logs(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / ".gitignore", tmp_path / ".gitignore")
    allowed = ["data/processed/handoff50/splits.json", "runs/handoff50/dev/metrics.json",
               "artifacts/handoff50/draft/head.joblib", "artifacts/handoff50/draft/smoke_samples.jsonl"]
    ignored = ["data/processed/handoff50/embeddings.npz", "runs/api_requests.jsonl",
               "artifacts/handoff50/index.faiss", "artifacts/encoder/model.safetensors",
               "data/raw/banking77/train.csv"]
    for name in allowed + ignored:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture", encoding="utf-8")
    result = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"], cwd=tmp_path,
                            input=b"\0".join(p.encode() for p in allowed + ignored) + b"\0",
                            capture_output=True, check=False)
    assert {p.decode() for p in result.stdout.split(b"\0") if p} == set(ignored)


def test_blank_source_rows_stay_aligned_but_cannot_be_used_for_fit_or_scoring(tmp_path, encoder):
    rows = [{"sample_id": "id_oos/test/0", "text": "", "label": "oos", "original_intent": "unknown",
             "split": "test", "group": "id_oos", "fit_eligible": False}]
    report = build_cache(rows, encoder, tmp_path, source_hashes={})
    with np.load(tmp_path / "embeddings.npz", allow_pickle=False) as data:
        assert data["sample_id"].tolist() == ["id_oos/test/0"]
        assert not data["valid_input"].any() and not data["fit_eligible"].any()
        assert not data["embeddings"].any()
    assert report["encoded_rows"] == 0 and report["invalid_input_ids"] == ["id_oos/test/0"]
