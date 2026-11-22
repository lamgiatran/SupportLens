"""The handoff checker must catch model/version mismatches, and reject mock APIs."""

from pathlib import Path
import json
import runpy

import httpx
import numpy as np
import pytest

from supportlens.banking import BankingPrediction


check_samples = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                   "scripts" / "check_banking_api.py"))["check_samples"]
SAMPLES = [{"input": {"question": question}} for question in ("card?", "transfer?", "balance?")]


class Pipeline:
    def predict(self, question, *, method):
        return BankingPrediction("card", 0.8, "accepted", "dev_gate", "real-head-v1")


def client_for(*, changed=None, mock=False):
    def handle(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"banking_model_ready": True,
                "banking_serving": {"state": "ready", "is_mock": mock}})
        return httpx.Response(200, json={"predicted_intent": "card", "score": 0.80000001,
            "decision": "accepted", "reason": "dev_gate", "artifact_version": "real-head-v1",
            "score_method": "H1", **(changed or {})})
    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))


def test_comparison_accepts_three_matching_outputs_with_float_tolerance():
    with client_for() as client:
        report = check_samples(Pipeline(), client, SAMPLES)
    assert report["status"] == "pass" and len(report["cases"]) == 3


def test_comparison_report_serializes_numpy_classifier_scores():
    class NumpyPipeline:
        def predict(self, question, *, method):
            return BankingPrediction("card", np.float32(0.8), "accepted", "dev_gate", "real-head-v1")

    with client_for() as client:
        report = check_samples(NumpyPipeline(), client, SAMPLES)
    assert json.loads(json.dumps(report))["status"] == "pass"


@pytest.mark.parametrize("changed", [{"artifact_version": "wrong-head"}, {"decision": "handoff"},
                                      {"score": 0.7}, {"score": None}, {"predicted_intent": "transfer"}])
def test_comparison_reports_output_or_artifact_mismatch(changed):
    with client_for(changed=changed) as client:
        report = check_samples(Pipeline(), client, SAMPLES)
    assert report["status"] == "fail"
    assert all(case["different_fields"] == list(changed) for case in report["cases"])


def test_comparison_rejects_mock_and_incorrect_sample_count():
    with client_for(mock=True) as client:
        with pytest.raises(ValueError, match="real Banking"):
            check_samples(Pipeline(), client, SAMPLES)
    with client_for() as client:
        with pytest.raises(ValueError, match="exactly 3"):
            check_samples(Pipeline(), client, SAMPLES[:2])
