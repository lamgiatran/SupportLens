"""Working document-retrieval slice. Missing model artifacts return explicit 503."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from .contracts import AssistRequest, AssistResponse, BankingRequest, Latency
from .evaluation import Gate
from .docs_gate import DocsGate
from .retrieval import BM25Chunks, BM25Documents
from .request_logging import RequestLogMiddleware


def create_app(retriever=None, docs_gate: Gate | None = None,
               log_path: str | Path | None = None, *, log_questions: bool = False,
               banking=None, retrievers: dict | None = None, docs_gates: dict | None = None) -> FastAPI:
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
        yield

    app = FastAPI(title="SupportLens", version="0.4.1", lifespan=lifespan)
    if log_path:
        app.add_middleware(RequestLogMiddleware, path=log_path, log_questions=log_questions)

    @app.get("/health")
    def health():
        return {"status": "ok", "docs_retrieval_ready": bool(registry), "docs_methods": sorted(registry),
                "docs_encoder_ready": all(e.ready for e, _ in encoders.values()) if encoders else None,
                "cold_encoder_warmup_ms": [e.warmup_ms for e, _ in encoders.values()],
                "docs_generator_ready": False, "banking_model_ready": banking is not None}

    @app.post("/v1/assist", response_model=AssistResponse)
    def assist(request: AssistRequest, http_request: Request):
        start = perf_counter()
        request_id = getattr(http_request.state, "request_id", None) or str(uuid4())
        http_request.state.response_record = {"mode": request.mode}
        if isinstance(request, BankingRequest):
            if banking is None:
                raise HTTPException(503, detail={"code": "handoff50_artifact_missing",
                                            "message": "Train and load the separate 50-label model first."})
            result = banking.predict(request.question, method=request.score_method)
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
        elapsed = (perf_counter() - start) * 1000
        reject = not evidence or (gate is not None and not gate.accepts(score))
        response = AssistResponse(
            request_id=request_id, mode="docs_qa", model_scope="multidoc2dial",
            artifact_version=active.version,
            decision="handoff" if reject else "retrieved",
            reason=("retrieval_below_dev_threshold" if reject else "retrieval_only_generator_not_loaded")
                   + (";gate_not_fitted" if gate is None else "")
                   + (";history_not_used_current_question_only" if request.history and not passage_mode else ""),
            evidence=evidence, score=score, score_method=score_method,
            latency=Latency(total_ms=elapsed, retrieve_ms=elapsed),
        )
        http_request.state.response_record = response.model_dump(exclude={"evidence", "answer"}) | query_audit
        return response

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
                      retrievers=registry, docs_gates=gates,
                      log_path=os.environ.get("SUPPORTLENS_LOG", "runs/api_requests.jsonl"),
                      log_questions=os.environ.get("SUPPORTLENS_LOG_QUESTIONS", "0") == "1")


app = app_from_env()
