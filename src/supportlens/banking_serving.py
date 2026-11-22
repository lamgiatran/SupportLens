"""Banking startup lifecycle. Requests never load models or retry failed startup."""

import importlib
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter

from .contracts import AssistResponse, Latency


# Inherit Uvicorn's configured handler and --log-level, including its default
# INFO level, without changing the root logger used by other libraries.
LOGGER = logging.getLogger("uvicorn.error.supportlens.banking_serving")
WARMUP_QUESTION = "My card payment was declined."


class BankingRuntime:
    def __init__(self, loader=None, *, is_mock=False):
        self.loader = loader
        self.is_mock = is_mock
        self.pipeline = None
        self.state = "pending" if loader is not None else "missing"
        self.error_type = None
        self.load_ms = self.warmup_ms = self.cold_start_ms = None
        self._startup_lock = Lock()

    @property
    def ready(self):
        return self.state == "ready"

    def snapshot(self):
        return {"state": self.state, "is_mock": self.is_mock,
                "load_ms": self.load_ms, "warmup_ms": self.warmup_ms,
                "cold_start_ms": self.cold_start_ms, "error_type": self.error_type}

    def start(self):
        # Idempotent even if two startup callers race. Failed attempts are retried
        # only by restarting the process, never by a user's request.
        with self._startup_lock:
            if self.state != "pending":
                return
            self.state = "loading"
            started = phase_started = perf_counter()
            phase = "load"
            try:
                pipeline = self.loader()
                if not callable(getattr(pipeline, "predict", None)):
                    raise TypeError("Banking factory must return a pipeline with predict(question, method=...)")
                load = getattr(pipeline, "load", None)
                if callable(load):
                    load()
                self.load_ms = (perf_counter() - phase_started) * 1000
                self.state = "warming"
                phase, phase_started = "warmup", perf_counter()
                result = pipeline.predict(WARMUP_QUESTION, method="H1")
                # Exercise encoder + head and validate the actual API contract
                # before publishing readiness. No independent classifier here.
                AssistResponse(request_id="startup-warmup", mode="banking_triage",
                               model_scope="handoff50", artifact_version=result.artifact_version,
                               decision=result.decision, reason=result.reason,
                               predicted_intent=result.predicted_intent, score=result.score,
                               score_method="H1", latency=Latency(total_ms=0))
                self.warmup_ms = (perf_counter() - phase_started) * 1000
                self.pipeline = pipeline
                self.state = "ready"
            except Exception as error:
                elapsed = (perf_counter() - phase_started) * 1000
                if phase == "load":
                    self.load_ms = elapsed
                else:
                    self.warmup_ms = elapsed
                self.error_type = type(error).__name__
                self.state = "failed"
                LOGGER.exception("Banking startup failed during %s", phase)
            finally:
                self.cold_start_ms = (perf_counter() - started) * 1000
                LOGGER.info("Banking startup: %s", self.snapshot())

    @property
    def unavailable_code(self):
        if self.state == "missing":
            return "handoff50_artifact_missing"
        if self.state == "failed":
            return "banking_startup_failed"
        return "banking_startup_not_complete"


def runtime_from_env(environ):
    """Explicit mock opt-in; a broken real factory never falls back to mock."""
    mode = environ.get("SUPPORTLENS_BANKING_MODE", "disabled")
    if mode == "disabled":
        return BankingRuntime()
    if mode == "mock":
        from .banking_mock import load_mock_pipeline
        return BankingRuntime(load_mock_pipeline, is_mock=True)
    if mode != "real":
        raise ValueError("SUPPORTLENS_BANKING_MODE must be disabled, mock or real")
    entrypoint = environ.get("SUPPORTLENS_BANKING_FACTORY", "")
    directory = environ.get("SUPPORTLENS_BANKING_ARTIFACTS", "")

    def load_real():
        if not directory or not Path(directory).is_dir():
            raise FileNotFoundError("Set SUPPORTLENS_BANKING_ARTIFACTS to the prepared local artifact directory")
        module_name, separator, function_name = entrypoint.partition(":")
        if not separator or not module_name or not function_name:
            raise ValueError("Set SUPPORTLENS_BANKING_FACTORY to module:function")
        factory = getattr(importlib.import_module(module_name), function_name)
        return factory(artifact_dir=Path(directory), local_files_only=True)

    return BankingRuntime(load_real)
