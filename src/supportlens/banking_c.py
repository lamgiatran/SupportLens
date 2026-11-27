"""Package C (Handoff-50) Confidence & Serving Factory supporting H1-H4.

Selected method for app: H1 (best AURC=0.14529 & highest coverage=51.70% at Risk<=10%).
Supports method in {'H1', 'H2', 'H3', 'H4'} so the UI selector works end-to-end.

Compatible with supportlens.banking_serving.runtime_from_env():
    SUPPORTLENS_BANKING_MODE=real
    SUPPORTLENS_BANKING_FACTORY=supportlens.banking_c:load_pipeline
    SUPPORTLENS_BANKING_ARTIFACTS=artifacts/handoff50/selected
"""

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp, softmax

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
    encoder._load()

    head = joblib.load(artifact_path / "head.joblib")
    fit_bank_path = artifact_path / "fit_bank.joblib"
    fit_bank = joblib.load(fit_bank_path) if fit_bank_path.exists() else None
    if fit_bank is not None:
        fit_bank = np.asarray(fit_bank, dtype=np.float32)

    classes = list(manifest["classes"])
    artifact_version = str(manifest["artifact_version"])
    temperature = float(manifest["h2_temperature"])
    knn_k = int(manifest.get("h3_knn_k", 10))
    gates = {
        m: Gate(
            method=m,
            artifact_version=artifact_version,
            threshold=manifest["thresholds"][m],
            fitted_split=manifest["fitted_split"],
        )
        for m in ("H1", "H2", "H3", "H4")
    }

    def predict_from_embedding(vector: np.ndarray, *, method: str = "H1") -> BankingPrediction:
        if method not in gates:
            raise ValueError(f"Unknown confidence method {method!r}; expected H1, H2, H3, or H4")
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        logits = head.decision_function(vec)[0]
        best_idx = int(np.argmax(logits))
        predicted_intent = str(classes[best_idx])

        if method == "H1":
            score = float(np.max(softmax(logits)))
        elif method == "H2":
            score = float(np.max(softmax(logits / temperature)))
        elif method == "H3":
            if fit_bank is None:
                raise FileNotFoundError("fit_bank.joblib is required for method='H3'")
            sims = (vec @ fit_bank.T)[0]
            top_k = np.partition(sims, -knn_k)[-knn_k:]
            score = float(np.mean(top_k))
        else:  # H4
            score = float(logsumexp(logits))

        if gates[method].accepts(score):
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
