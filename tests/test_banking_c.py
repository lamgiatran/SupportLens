"""Unit tests for Package C (banking_c) supporting H1-H4 and BankingRuntime/API."""

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


def _make_c_artifact(artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    enc_cfg = EncoderConfig("offline-test", "d" * 40, dimension=4)
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
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.asarray(["card_arrival", "change_pin", "lost_or_stolen_card"])
    clf = sklearn_lr.LogisticRegression(C=10.0, max_iter=200, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, artifact_dir / "head.joblib")
    joblib.dump(X.astype(np.float16), artifact_dir / "fit_bank.joblib")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({
            "model_scope": "handoff50",
            "artifact_version": "handoff50-pkgc-test",
            "factory": "supportlens.banking_c:load_pipeline",
            "selected_method": "H1",
            "selected_threshold": 0.40,
            "thresholds": {"H1": 0.40, "H2": 0.45, "H3": 0.50, "H4": 1.0},
            "h2_temperature": 0.75,
            "h3_knn_k": 2,
            "fitted_split": "dev",
            "encoder_fingerprint": enc_cfg.fingerprint,
            "classes": list(clf.classes_),
        }),
        encoding="utf-8",
    )
    return artifact_dir


def test_banking_c_factory_supports_h1_to_h4_with_api(tmp_path, monkeypatch):
    artifact_dir = _make_c_artifact(tmp_path / "selected")
    monkeypatch.setattr(
        "supportlens.banking_c.BankingEncoder",
        lambda cfg, local_files_only=True: BankingEncoder(cfg, backend=_OfflineBackend(), local_files_only=local_files_only),
    )

    runtime = runtime_from_env({
        "SUPPORTLENS_BANKING_MODE": "real",
        "SUPPORTLENS_BANKING_FACTORY": "supportlens.banking_c:load_pipeline",
        "SUPPORTLENS_BANKING_ARTIFACTS": str(artifact_dir),
    })
    with TestClient(create_app(banking_runtime=runtime)) as client:
        for method in ("H1", "H2", "H3", "H4"):
            res = client.post("/v1/assist", json={
                "mode": "banking_triage",
                "model_scope": "handoff50",
                "question": "How do I change my pin?",
                "score_method": method,
                "history": [],
            })
            assert res.status_code == 200
            body = res.json()
            assert body["score_method"] == method
            assert body["artifact_version"] == "handoff50-pkgc-test"
            assert body["decision"] in {"accepted", "handoff"}
