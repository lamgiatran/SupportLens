"""Cau owns the Banking pipeline. Cache building and serving call ONE encode().

The frozen encoder does not fit on any Banking labels. The separate 50-label
head and dev-fitted thresholds are supplied by Cau, not fabricated here.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class EncoderConfig:
    model_id: str
    revision: str
    max_seq_length: int = 256
    dimension: int = 384
    normalize_embeddings: bool = True
    text_policy: str = "strip_current_question_only_v1"

    def __post_init__(self):
        if (len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision)
                or self.max_seq_length < 3 or self.dimension < 1):
            raise ValueError("pin a full encoder commit and valid dimensions/budget")
        if self.text_policy != "strip_current_question_only_v1":
            raise ValueError("unsupported text policy")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class BankingEncoder:
    def __init__(self, config: EncoderConfig, *, device: str = "cpu", backend=None,
                 local_files_only: bool = False):
        self.config, self.device = config, device
        self._backend = backend
        self.local_files_only = local_files_only

    def _load(self):
        if self._backend is None:
            from sentence_transformers import SentenceTransformer
            self._backend = SentenceTransformer(
                self.config.model_id, revision=self.config.revision, device=self.device,
                trust_remote_code=False, local_files_only=self.local_files_only)
        self._backend.max_seq_length = self.config.max_seq_length
        return self._backend

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str) or batch_size < 1:
            raise ValueError("pass a sequence of strings and a positive batch size")
        values = list(texts)
        if any(not isinstance(t, str) or not t.strip() for t in values):
            raise ValueError("Banking inputs must be non-blank strings")
        if not values:
            return np.empty((0, self.config.dimension), dtype=np.float32)
        result = np.asarray(self._load().encode(
            [t.strip() for t in values], batch_size=batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if result.shape != (len(values), self.config.dimension) or not np.isfinite(result).all():
            raise ValueError("encoder returned invalid dimensions or non-finite vectors")
        if self.config.normalize_embeddings and not np.allclose(
                np.linalg.norm(result, axis=1), 1, atol=1e-4):
            raise ValueError("encoder vectors do not satisfy L2 normalization")
        return result


@dataclass(frozen=True)
class BankingPrediction:
    predicted_intent: str
    score: float
    decision: Literal["accepted", "handoff"]
    reason: str
    artifact_version: str


class BankingPipeline:
    def __init__(self, encoder: BankingEncoder,
                 predict_from_embedding: Callable[..., BankingPrediction], *,
                 expected_encoder_fingerprint: str):
        if expected_encoder_fingerprint != encoder.config.fingerprint:
            raise ValueError("head and live encoder do not match")
        self.encoder = encoder
        self.predict_from_embedding = predict_from_embedding

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        return self.encoder.encode(texts, batch_size=batch_size)

    def predict(self, question: str, *, method: str = "H1") -> BankingPrediction:
        if method not in {"H1", "H2", "H3", "H4"}:
            raise ValueError("unknown confidence method")
        return self.predict_from_embedding(self.encode([question])[0], method=method)
