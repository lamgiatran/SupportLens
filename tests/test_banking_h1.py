"""Unit tests for Package B (banking_h1) factory and BankingRuntime/API integration."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from supportlens.api import create_app
from supportlens.banking import BankingEncoder, EncoderConfig
from supportlens.banking_serving import runtime_from_env

joblib = pytest.importorskip("joblib")
sklearn_lr = pytest.importorskip("sklearn.linear_model")


class _OfflineBackend:
    def encode(self, texts, **kwargs):
        arr = np.asarray([[1.0, float(len(t)), 2.0, 3.0] for t in texts], dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def _make_h1_artifact(artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    enc_cfg = EncoderConfig("offline-test", "c" * 40, dimension=4)
    (artifact_dir / "encoder_config.json").write_text(
        json.dumps({
            "model_id": enc_cfg.model_id,
            "revision": enc_cfg.revision,
            "max_seq_length": enc_cfg.max_seq_length,
            "dimension": enc_cfg.dimension,
            "normalize_embeddings": enc_cfg.normalize_embeddings,
            "text_policy": enc_cfg.text_policy,
        }),
        encoding="utf-8",
    )
    X = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    y = np.asarray(["card_arrival", "change_pin", "lost_or_stolen_card"])
    clf = sklearn_lr.LogisticRegression(C=10.0, max_iter=200, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, artifact_dir / "head.joblib")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({
            "model_scope": "handoff50",
            "artifact_version": "handoff50-lr-h1-test",
            "factory": "supportlens.banking_h1:load_pipeline",
            "method": "H1",
            "selected_C": 10.0,
            "threshold": 0.40,
            "fitted_split": "dev",
            "encoder_fingerprint": enc_cfg.fingerprint,
            "classes": list(clf.classes_),
        }),
        encoding="utf-8",
    )
    return artifact_dir


def test_banking_h1_factory_works_with_runtime_from_env_and_api(tmp_path, monkeypatch):
    artifact_dir = _make_h1_artifact(tmp_path / "h1")
    monkeypatch.setattr(
        "supportlens.banking_h1.BankingEncoder",
        lambda cfg, local_files_only=True: BankingEncoder(cfg, backend=_OfflineBackend(), local_files_only=local_files_only),
    )

    runtime = runtime_from_env({
        "SUPPORTLENS_BANKING_MODE": "real",
        "SUPPORTLENS_BANKING_FACTORY": "supportlens.banking_h1:load_pipeline",
        "SUPPORTLENS_BANKING_ARTIFACTS": str(artifact_dir),
    })
    with TestClient(create_app(banking_runtime=runtime)) as client:
        health = client.get("/health").json()
        assert health["banking_model_ready"] is True
        assert health["banking_serving"]["is_mock"] is False
        assert health["banking_serving"]["state"] == "ready"

        res = client.post("/v1/assist", json={
            "mode": "banking_triage",
            "model_scope": "handoff50",
            "question": "My card has not arrived yet",
            "score_method": "H1",
            "history": [],
        })
        assert res.status_code == 200
        body = res.json()
        assert body["artifact_version"] == "handoff50-lr-h1-test"
        assert body["decision"] in {"accepted", "handoff"}
        assert 0.0 <= body["score"] <= 1.0

        with pytest.raises(ValueError, match="H1"):
            runtime.pipeline.predict("test", method="H2")
