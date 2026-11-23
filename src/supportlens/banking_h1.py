"""Package B (Handoff-50) Logistic Regression + H1 serving factory.

Compatible with supportlens.banking_serving.runtime_from_env():
    SUPPORTLENS_BANKING_MODE=real
    SUPPORTLENS_BANKING_FACTORY=supportlens.banking_h1:load_pipeline
    SUPPORTLENS_BANKING_ARTIFACTS=artifacts/handoff50/h1
"""

import json
from pathlib import Path

import joblib
import numpy as np

from .banking import BankingEncoder, BankingPipeline, BankingPrediction, EncoderConfig
from .evaluation import Gate


def load_pipeline(*, artifact_dir: str | Path, local_files_only: bool = True) -> BankingPipeline:
    """Factory matching module:function(artifact_dir, local_files_only) for banking_serving."""
    artifact_path = Path(artifact_dir)
    if not artifact_path.is_dir():
        raise FileNotFoundError(f"Handoff-50 artifact directory not found: {artifact_path}")

    manifest = json.loads((artifact_path / "manifest.json").read_text(encoding="utf-8"))
    encoder_config = EncoderConfig.load(artifact_path / "encoder_config.json")
    if encoder_config.fingerprint != manifest["encoder_fingerprint"]:
        raise ValueError("encoder_config.json fingerprint does not match manifest.json")

    encoder = BankingEncoder(encoder_config, local_files_only=local_files_only)
    # Eagerly load encoder weights during startup so requests never trigger lazy downloads.
    encoder._load()

    head = joblib.load(artifact_path / "head.joblib")
    classes = list(manifest["classes"])
    artifact_version = str(manifest["artifact_version"])
    gate = Gate(
        method="H1",
        artifact_version=artifact_version,
        threshold=manifest["threshold"],
        fitted_split=manifest["fitted_split"],
    )

    def predict_from_embedding(vector: np.ndarray, *, method: str = "H1") -> BankingPrediction:
        if method != "H1":
            raise ValueError(f"Package B head only supports method='H1', got {method!r}")
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        probs = head.predict_proba(vec)[0]
        best_idx = int(np.argmax(probs))
        score = float(probs[best_idx])
        predicted_intent = str(classes[best_idx])
        if gate.accepts(score):
            return BankingPrediction(
                predicted_intent=predicted_intent,
                score=score,
                decision="accepted",
                reason="dev_gate_passed",
                artifact_version=artifact_version,
            )
        return BankingPrediction(
            predicted_intent=predicted_intent,
            score=score,
            decision="handoff",
            reason="low_confidence_handoff",
            artifact_version=artifact_version,
        )

    return BankingPipeline(
        encoder,
        predict_from_embedding,
        expected_encoder_fingerprint=manifest["encoder_fingerprint"],
    )
