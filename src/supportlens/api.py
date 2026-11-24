"""SupportLens API: Docs-QA retrieval/generation and Banking triage. Missing artifacts return explicit 503."""

import json
import logging
import os
from threading import Event
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from .contracts import AssistRequest, AssistResponse, BankingRequest, Latency
from .evaluation import Gate
from .docs_gate import DocsGate
from .retrieval import BM25Chunks, BM25Documents
from .request_logging import RequestLogMiddleware
from .banking_serving import BankingRuntime, runtime_from_env
from .docs_generator import generator_from_env


def create_app(retriever=None, docs_gate: Gate | None = None,
               log_path: str | Path | None = None, *, log_questions: bool = False,
               banking=None, banking_loader=None, banking_runtime=None,
               retrievers: dict | None = None, docs_gates: dict | None = None,
               generator=None) -> FastAPI:
    if sum(value is not None for value in (banking, banking_loader, banking_runtime)) > 1:
        raise ValueError("Supply only one of banking, banking_loader or banking_runtime")
    runtime = banking_runtime if banking_runtime is not None else BankingRuntime(
        banking_loader if banking_loader is not None else (lambda: banking) if banking is not None else None)
    registry = dict(retrievers or {})
    if retriever is not None:
        registry.setdefault(getattr(retriever, "method", "R1"), retriever)
    gates = dict(docs_gates or {})
    if docs_gate is not None:
        gates[docs_gate.method] = docs_gate
    encoders = {}
    for value in registry.values():
        dense = getattr(value, "dense", value)
        encoder = getattr(dense, "encoder", None)
        if encoder is not None:
            encoders[id(encoder)] = (encoder, value.tokenizer)

    @asynccontextmanager
    async def lifespan(app):
        # No traffic before model loading, tokenizer audit and warm-up.
        # R2 and R3 share a single encoder; first request never downloads weights.
        for encoder, tokenizer in encoders.values():
            encoder.warmup(tokenizer)
        runtime.start()
        if generator is not None:
            generator.start()
        yield

    app = FastAPI(title="SupportLens", version="0.4.1", lifespan=lifespan)
    app.state.banking_runtime = runtime
    if log_path:
        app.add_middleware(RequestLogMiddleware, path=log_path, log_questions=log_questions)

    @app.get("/health")
    def health():
        return {"status": "ok", "docs_retrieval_ready": bool(registry), "docs_methods": sorted(registry),
                "docs_encoder_ready": all(e.ready for e, _ in encoders.values()) if encoders else None,
                "cold_encoder_warmup_ms": [e.warmup_ms for e, _ in encoders.values()],
                "docs_generator_ready": bool(generator is not None and generator.ready),
                "docs_generator": generator.snapshot() if generator is not None else {"state": "disabled"},
                "banking_model_ready": runtime.ready,
                "banking_serving": runtime.snapshot()}

    @app.get("/ready/banking")
    def banking_readiness():
        if not runtime.ready:
            raise HTTPException(503, detail={"code": runtime.unavailable_code,
                                             "banking_serving": runtime.snapshot()})
        return {"status": "ready", "banking_serving": runtime.snapshot()}

    def banking_response(request: BankingRequest, request_id: str, http_request: Request, start: float):
        if not runtime.ready:
            code = runtime.unavailable_code
            http_request.state.response_record.update(
                error_code=code, error_type=runtime.error_type, banking_state=runtime.state)
            logging.getLogger(__name__).warning(
                "Banking unavailable request_id=%s code=%s startup_error_type=%s",
                request_id, code, runtime.error_type)
            message = ("Train and load the separate 50-label model first." if runtime.state == "missing"
                       else "Banking is not ready; check startup logs and restart after fixing the model configuration.")
            raise HTTPException(503, detail={"code": code, "message": message},
                                headers={"X-Request-ID": request_id})
        result = runtime.pipeline.predict(request.question, method=request.score_method)
        elapsed = (perf_counter() - start) * 1000
        response = AssistResponse(
            request_id=request_id, mode=request.mode, model_scope="handoff50",
            artifact_version=result.artifact_version, decision=result.decision,
            reason=result.reason + (";history_not_used_current_question_only" if request.history else ""),
            predicted_intent=result.predicted_intent, score=result.score,
            score_method=request.score_method,
            latency=Latency(total_ms=elapsed, classify_ms=elapsed))
        http_request.state.response_record = response.model_dump(exclude={"evidence", "answer"})
        return response

    def docs_retrieval(request):
        """Validate artifacts and retrieve passages. Raises 503 before any stream starts."""
        active = registry.get(request.retriever)
        gate = gates.get(request.retriever)
        if active is None:
            raise HTTPException(503, detail={"code": "retriever_not_loaded", "retriever": request.retriever})
        dense = getattr(active, "dense", active)
        if hasattr(dense, "encoder") and not dense.encoder.ready:
            raise HTTPException(503, detail={"code": "encoder_startup_not_complete"})
        if gate and (gate.artifact_version != active.version or gate.method != request.retriever):
            raise HTTPException(503, detail={"code": "threshold_artifact_mismatch"})
        if gate and gate.threshold is not None and (
                getattr(gate, "score_name", None) != getattr(active, "score_name", None)
                or getattr(gate, "target_k", None) != 3):
            raise HTTPException(503, detail={"code": "threshold_score_policy_mismatch"})
        started = perf_counter()
        query_audit = {}
        passage_mode = hasattr(active, "prepare_query")
        if passage_mode:
            query = active.prepare_query(request.question, [t.model_dump() for t in request.history])
            query_text = query.text
            query_audit = {"query_original_tokens": query.original_tokens, "query_used_tokens": query.used_tokens,
                           "query_truncated": query.truncated,
                           "current_question_truncated": query.current_question_truncated}
        else:
            query_text = request.question
        if hasattr(active, "retrieve"):
            evidence, score = active.retrieve(query_text, k=5)
            score_method = active.score_name
        else:
            evidence = active.search(query_text, k=5)
            score = evidence[0].score if evidence else None
            score_method = "R1_document_bm25"
        reject = not evidence or (gate is not None and not gate.accepts(score))
        suffix = (("" if gate is not None else ";gate_not_fitted")
                  + (";history_not_used_current_question_only" if request.history and not passage_mode else ""))
        return {"active": active, "evidence": evidence, "score": score, "score_method": score_method,
                "reject": reject, "suffix": suffix, "query_audit": query_audit,
                "retrieve_ms": (perf_counter() - started) * 1000}

    def docs_response(request_id, retrieval, start, result=None, *, generator_unavailable=False):
        """Single place that turns retrieval (+ optional generation) into the response contract."""
        answer, citations, cache_status, ttft_ms, generate_ms = None, [], "not_used", None, None
        artifact_version = retrieval["active"].version
        if retrieval["reject"]:
            decision, reason = "handoff", "retrieval_below_dev_threshold"
        elif result is not None:
            decision, reason = result.decision()
            if decision == "answer":
                answer, citations = result.answer, result.citations
            cache_status, ttft_ms, generate_ms = result.cache_status, result.ttft_ms, result.generate_ms
            artifact_version += "+" + generator.version
        else:
            decision = "retrieved"
            reason = ("retrieval_only_generator_not_ready" if generator_unavailable
                      else "retrieval_only_generator_not_loaded")
        return AssistResponse(
            request_id=request_id, mode="docs_qa", model_scope="multidoc2dial",
            artifact_version=artifact_version, decision=decision, reason=reason + retrieval["suffix"],
            answer=answer, evidence=retrieval["evidence"], citations=citations,
            score=retrieval["score"], score_method=retrieval["score_method"], cache_status=cache_status,
            latency=Latency(total_ms=(perf_counter() - start) * 1000, retrieve_ms=retrieval["retrieve_ms"],
                            generate_ms=generate_ms, ttft_ms=ttft_ms))

    def record(http_request, response, retrieval=None, result=None):
        value = response.model_dump(exclude={"evidence", "answer"})
        if retrieval is not None:
            value |= retrieval["query_audit"]
        if result is not None:
            value |= {"generator_status": result.status, "generator_error_type": result.error_type,
                      "invalid_citations": result.invalid_citations}
        http_request.state.response_record = value

    def begin(http_request, mode):
        request_id = getattr(http_request.state, "request_id", None) or str(uuid4())
        http_request.state.request_id = request_id
        http_request.state.response_record = {"mode": mode}
        return request_id

    def wants_generation(retrieval):
        return generator is not None and generator.ready and not retrieval["reject"]

    @app.post("/v1/assist", response_model=AssistResponse)
    def assist(request: AssistRequest, http_request: Request):
        start = perf_counter()
        request_id = begin(http_request, request.mode)
        if isinstance(request, BankingRequest):
            return banking_response(request, request_id, http_request, start)
        retrieval = docs_retrieval(request)
        result = None
        if wants_generation(retrieval):
            result = generator.generate(request.question, retrieval["evidence"],
                                        history=[turn.model_dump() for turn in request.history],
                                        retrieval_version=retrieval["active"].version)
        response = docs_response(request_id, retrieval, start, result,
                                 generator_unavailable=generator is not None and not generator.ready)
        record(http_request, response, retrieval, result)
        return response

    @app.post("/v1/assist/stream")
    async def assist_stream(request: AssistRequest, http_request: Request):
        """NDJSON events: start, delta*, done | error. See docs/RUN_GENERATION.md for the schema."""
        start = perf_counter()
        request_id = begin(http_request, request.mode)
        headers = {"X-Request-ID": request_id, "Cache-Control": "no-cache"}

        def line(value):
            return json.dumps(value, ensure_ascii=False) + "\n"

        if isinstance(request, BankingRequest):
            response = banking_response(request, request_id, http_request, start)

            async def banking_events():
                yield line({"event": "start", "request_id": request_id, "mode": request.mode, "evidence": []})
                yield line({"event": "done", "response": response.model_dump(mode="json")})
            return StreamingResponse(banking_events(), media_type="application/x-ndjson", headers=headers)

        retrieval = docs_retrieval(request)
        cancel = Event()

        async def docs_events():
            result = None
            try:
                yield line({"event": "start", "request_id": request_id, "mode": request.mode,
                            "evidence": [e.model_dump(mode="json") for e in retrieval["evidence"]],
                            "retrieve_ms": retrieval["retrieve_ms"], "generating": wants_generation(retrieval)})
                if wants_generation(retrieval):
                    events = generator.stream(request.question, retrieval["evidence"], cancel=cancel,
                                              history=[turn.model_dump() for turn in request.history],
                                              retrieval_version=retrieval["active"].version)
                    async for event in iterate_in_threadpool(events):
                        if await http_request.is_disconnected():
                            cancel.set()
                            return
                        if event["event"] == "delta":
                            yield line({"event": "delta", "text": event["text"]})
                        else:
                            result = event["result"]
                response = docs_response(request_id, retrieval, start, result,
                                         generator_unavailable=generator is not None and not generator.ready)
                record(http_request, response, retrieval, result)
                yield line({"event": "done", "response": response.model_dump(mode="json")})
            except Exception as error:  # report, never hang the client
                logging.getLogger(__name__).exception("Docs stream failed request_id=%s", request_id)
                yield line({"event": "error", "request_id": request_id, "code": "stream_failed",
                            "error_type": type(error).__name__})
            finally:
                cancel.set()

        return StreamingResponse(docs_events(), media_type="application/x-ndjson", headers=headers)

    return app


def app_from_env():
    corpus = os.environ.get("SUPPORTLENS_CORPUS")
    bundle = os.environ.get("SUPPORTLENS_DOCS_BUNDLE")
    retriever = (BM25Chunks.from_bundle(bundle, os.environ.get("SUPPORTLENS_DOCS_TOKENIZER", "data/tokenizers/minilm/tokenizer.json"))
                 if bundle else BM25Documents.from_jsonl(corpus) if corpus else None)
    gate_path = os.environ.get("SUPPORTLENS_DOCS_GATE")
    gate = DocsGate(**json.loads(Path(gate_path).read_text(encoding="utf-8"))["gate"]) if gate_path else None
    registry = {"R1": retriever} if retriever else {}
    dense_dir = os.environ.get("SUPPORTLENS_DOCS_DENSE")
    if dense_dir:
        if not isinstance(retriever, BM25Chunks):
            raise ValueError("dense retrieval requires a verified passage bundle")
        from .dense import DenseEncoder, DenseEncoderConfig, DenseChunks, HybridChunks, load_corpus_cache
        encoder = DenseEncoder(DenseEncoderConfig.load(os.environ.get("SUPPORTLENS_DOCS_ENCODER", "configs/docs_encoder.json")),
                               local_files_only=True, device=os.environ.get("SUPPORTLENS_ENCODER_DEVICE", "cpu"))
        matrix, _ = load_corpus_cache(retriever, encoder, dense_dir)
        registry["R2"] = DenseChunks(retriever, encoder, matrix)
        registry["R3"] = HybridChunks(retriever, registry["R2"])
    gate_dir = os.environ.get("SUPPORTLENS_DOCS_GATES")
    gates = {}
    if gate_dir:
        if gate_path:
            raise ValueError("use a single gate file OR a gate directory")
        for method in registry:
            gates[method] = DocsGate(**json.loads((Path(gate_dir) / (method + ".json")).read_text(encoding="utf-8"))["gate"])
    return create_app(retriever, docs_gate=gate,
                      banking_runtime=runtime_from_env(os.environ),
                      generator=generator_from_env(os.environ),
                      retrievers=registry, docs_gates=gates,
                      log_path=os.environ.get("SUPPORTLENS_LOG", "runs/api_requests.jsonl"),
                      log_questions=os.environ.get("SUPPORTLENS_LOG_QUESTIONS", "0") == "1")


app = app_from_env()
