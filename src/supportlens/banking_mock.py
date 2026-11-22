"""Explicit local serving demo; synthetic vectors/head, never a trained model."""

import numpy as np

from .banking import BankingEncoder, BankingPipeline, BankingPrediction, EncoderConfig


class MockBackend:
    def encode(self, texts, **kwargs):
        return np.tile(np.asarray([[1, 0, 0, 0]], dtype=np.float32), (len(texts), 1))


def load_mock_pipeline():
    encoder = BankingEncoder(
        EncoderConfig("MOCK-no-pretrained-model", "0" * 40, dimension=4),
        backend=MockBackend(), local_files_only=True)

    def head(vector, *, method):
        return BankingPrediction("MOCK_INTENT", 0.0, "handoff",
                                 "mock_development_only_no_real_prediction", "MOCK-banking-serving-v1")

    return BankingPipeline(encoder, head, expected_encoder_fingerprint=encoder.config.fingerprint)
