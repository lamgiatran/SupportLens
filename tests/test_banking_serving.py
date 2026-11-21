"""Serving acceptance checks with offline doubles, no Cau model required."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Event, Lock
from time import sleep
from types import ModuleType

from fastapi.testclient import TestClient
import numpy as np
import pytest

from supportlens.api import create_app
from supportlens.banking import BankingEncoder, BankingPipeline, BankingPrediction, EncoderConfig
from supportlens.banking_serving import BankingRuntime, WARMUP_QUESTION, runtime_from_env
from supportlens.retrieval import BM25Documents


PAYLOAD = {"mode": "banking_triage", "question": "Please replace my card", "score_method": "H1"}


def test_default_uvicorn_logging_emits_startup_timings_once():
    # A separate process keeps Uvicorn's dictConfig from altering pytest's
    # capture handlers. Configure the same defaults as the documented CLI.
    script = """
import logging
import sys
sys.path.insert(0, 'src')
from uvicorn import Config
from supportlens.banking_serving import runtime_from_env
Config('supportlens.api:app')
runtime = runtime_from_env({'SUPPORTLENS_BANKING_MODE': 'mock'})
runtime.start()
runtime.start()
assert runtime.ready
assert logging.getLogger().level == logging.WARNING
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True, timeout=15)
    assert result.stderr.count("Banking startup:") == 1
    assert "'state': 'ready'" in result.stderr
    for key in ("load_ms", "warmup_ms", "cold_start_ms"):
        assert f"'{key}':" in result.stderr
        assert f"'{key}': None" not in result.stderr


class ProbePipeline:
    def __init__(self, failure=None):
        self.failure = failure
        self.loads = 0
        self.questions = []

    def load(self):
        self.loads += 1
        if self.failure == "load":
            raise FileNotFoundError("offline head missing")

    def predict(self, question, *, method):
        self.questions.append((question, method))
        if self.failure == "warmup":
            raise RuntimeError("warm-up failed")
        if self.failure == "contract":
            return BankingPrediction("card", float("nan"), "accepted", "test", "test-v1")
        return BankingPrediction("card", 0.8, "accepted", "test_head", "test-v1")


def test_no_request_can_trigger_startup_and_health_waits_for_lifespan():
    calls = []
    app = create_app(banking_loader=lambda: calls.append(1) or ProbePipeline())
    client = TestClient(app)  # Deliberately do not run lifespan.
    assert client.get("/health").json()["banking_model_ready"] is False
    response = client.post("/v1/assist", json=PAYLOAD)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "banking_startup_not_complete"
    assert client.get("/ready/banking").status_code == 503
    assert not calls


def test_startup_warms_head_before_ready_and_requests_reuse_pipeline():
    pipeline, calls = ProbePipeline(), []
    app = create_app(banking_loader=lambda: calls.append(1) or pipeline)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["banking_model_ready"] is True
        assert pipeline.questions == [(WARMUP_QUESTION, "H1")]
        assert client.get("/ready/banking").status_code == 200
        timings = health["banking_serving"]
        assert timings["state"] == "ready" and timings["error_type"] is None
        assert all(timings[key] >= 0 for key in ("load_ms", "warmup_ms", "cold_start_ms"))
        assert timings["cold_start_ms"] >= timings["load_ms"] + timings["warmup_ms"]
        for _ in range(3):
            response = client.post("/v1/assist", json=PAYLOAD)
            assert response.status_code == 200
            assert response.json()["predicted_intent"] == "card"
            assert response.json()["latency"]["classify_ms"] >= 0
        assert client.get("/health").json()["banking_serving"] == timings
    assert calls == [1] and pipeline.loads == 1 and len(pipeline.questions) == 4


def test_readiness_is_false_while_warmup_is_in_progress():
    entered, release = Event(), Event()
    pipeline = ProbePipeline()
    original = pipeline.predict

    def predict(question, *, method):
        entered.set()
        assert release.wait(5)
        return original(question, method=method)

    pipeline.predict = predict
    runtime = BankingRuntime(lambda: pipeline)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runtime.start)
        try:
            assert entered.wait(5)
            assert not runtime.ready and runtime.state == "warming"
            assert runtime.pipeline is None
        finally:
            release.set()
        future.result(timeout=5)
    assert runtime.ready


def test_two_startup_callers_only_load_and_warm_once():
    barrier, calls, pipeline = Barrier(2), [], ProbePipeline()

    def loader():
        calls.append(1)
        sleep(0.03)  # Release GIL while the second caller attempts startup.
        return pipeline

    runtime = BankingRuntime(loader)

    def start():
        barrier.wait(timeout=5)
        runtime.start()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(start) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert runtime.ready and calls == [1]
    assert pipeline.loads == 1 and pipeline.questions == [(WARMUP_QUESTION, "H1")]


def test_concurrent_lazy_encoder_callers_construct_one_configured_backend(monkeypatch):
    config = EncoderConfig("offline-double", "a" * 40, dimension=4)
    encoder = BankingEncoder(config, local_files_only=True)
    barrier, counter_lock, calls = Barrier(8), Lock(), []

    class Backend:
        def encode(self, texts, **kwargs):
            assert self.max_seq_length == config.max_seq_length
            return np.tile([[1, 0, 0, 0]], (len(texts), 1))

    backend = Backend()

    def factory():
        with counter_lock:
            calls.append(1)
        sleep(0.03)
        return backend

    monkeypatch.setattr(encoder, "_create_backend", factory)

    def encode():
        barrier.wait(timeout=5)
        return encoder.encode(["card?"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(encode) for _ in range(8)]
        for future in futures:
            np.testing.assert_array_equal(future.result(timeout=5), [[1, 0, 0, 0]])
    assert calls == [1] and encoder._backend is backend


@pytest.mark.parametrize("failure", ["factory", "load", "warmup", "contract"])
def test_failed_startup_is_unready_logged_and_never_retried_by_requests(
        tmp_path, rows, caplog, failure):
    calls, pipeline = [], ProbePipeline(failure)

    def loader():
        calls.append(1)
        if failure == "factory":
            raise FileNotFoundError("missing weights")
        return pipeline

    log_path = tmp_path / "requests.jsonl"
    app = create_app(BM25Documents(rows), banking_loader=loader, log_path=log_path)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert not health["banking_model_ready"]
        startup = health["banking_serving"]
        assert startup["state"] == "failed" and startup["error_type"]
        assert startup["cold_start_ms"] >= 0 and startup["load_ms"] >= 0
        assert client.get("/ready/banking").status_code == 503
        for _ in range(2):
            response = client.post("/v1/assist", json=PAYLOAD)
            assert response.status_code == 503
            assert response.json()["detail"]["code"] == "banking_startup_failed"
            assert response.headers["x-request-id"] in caplog.text
            assert "," not in response.headers["x-request-id"]
        # Banking failure must leave Docs-QA available.
        assert client.post("/v1/assist", json={"mode": "docs_qa", "question": "uniqueaddress"}).status_code == 200
        app.state.banking_runtime.start()
        assert app.state.banking_runtime.pipeline is None
    assert calls == [1]
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    errors = [record for record in records if record.get("mode") == "banking_triage"]
    assert len(errors) == 2
    assert all(record["status_code"] == 503 and record["error_code"] == "banking_startup_failed"
               and record["error_type"] == startup["error_type"] for record in errors)


def test_disabled_banking_keeps_missing_artifact_contract():
    with TestClient(create_app(banking_runtime=runtime_from_env({}))) as client:
        health = client.get("/health").json()
        assert not health["banking_model_ready"]
        assert health["banking_serving"]["state"] == "missing"
        response = client.post("/v1/assist", json=PAYLOAD)
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "handoff50_artifact_missing"


def test_mock_requires_explicit_opt_in_and_is_marked_in_health_and_response():
    runtime = runtime_from_env({"SUPPORTLENS_BANKING_MODE": "mock"})
    with TestClient(create_app(banking_runtime=runtime)) as client:
        health = client.get("/health").json()
        assert health["banking_model_ready"] and health["banking_serving"]["is_mock"]
        result = client.post("/v1/assist", json=PAYLOAD).json()
        assert result["artifact_version"].startswith("MOCK-")
        assert result["decision"] == "handoff" and result["predicted_intent"] == "MOCK_INTENT"
        assert "mock_development_only" in result["reason"]


def test_real_factory_receives_local_artifacts_and_offline_policy(tmp_path, monkeypatch):
    module = ModuleType("offline_cau_factory")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return ProbePipeline()

    module.load_pipeline = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = runtime_from_env({"SUPPORTLENS_BANKING_MODE": "real",
                                "SUPPORTLENS_BANKING_ARTIFACTS": str(tmp_path),
                                "SUPPORTLENS_BANKING_FACTORY": "offline_cau_factory:load_pipeline"})
    assert not calls
    runtime.start()
    assert runtime.ready and not runtime.is_mock
    assert calls == [{"artifact_dir": tmp_path, "local_files_only": True}]


@pytest.mark.parametrize("entrypoint", ["", "bad-syntax", "missing_cau_module:load_pipeline"])
def test_bad_real_factory_fails_without_mock_fallback(tmp_path, entrypoint):
    runtime = runtime_from_env({"SUPPORTLENS_BANKING_MODE": "real",
                                "SUPPORTLENS_BANKING_ARTIFACTS": str(tmp_path),
                                "SUPPORTLENS_BANKING_FACTORY": entrypoint})
    runtime.start()
    assert not runtime.ready and not runtime.is_mock and runtime.state == "failed"


def test_missing_real_artifact_directory_fails_without_mock_fallback(tmp_path):
    runtime = runtime_from_env({"SUPPORTLENS_BANKING_MODE": "real",
                                "SUPPORTLENS_BANKING_ARTIFACTS": str(tmp_path / "missing")})
    runtime.start()
    assert not runtime.ready and runtime.error_type == "FileNotFoundError"


def test_serving_pipeline_refuses_a_backend_that_could_download(monkeypatch):
    encoder = BankingEncoder(EncoderConfig("unused", "a" * 40, dimension=4))
    calls = []
    monkeypatch.setattr(encoder, "_create_backend", lambda: calls.append(1))
    pipeline = BankingPipeline(encoder, lambda *a: None,
                               expected_encoder_fingerprint=encoder.config.fingerprint)
    runtime = BankingRuntime(lambda: pipeline)
    runtime.start()
    assert runtime.state == "failed" and runtime.error_type == "ValueError" and not calls


def test_default_backend_factory_forwards_local_files_only(monkeypatch):
    module, calls = ModuleType("sentence_transformers"), []
    module.SentenceTransformer = lambda *args, **kwargs: calls.append((args, kwargs)) or object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    encoder = BankingEncoder(EncoderConfig("local-encoder", "b" * 40), local_files_only=True)
    encoder._create_backend()
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["trust_remote_code"] is False


def test_conflicting_banking_sources_are_rejected():
    with pytest.raises(ValueError, match="only one"):
        create_app(banking=ProbePipeline(), banking_loader=ProbePipeline)
