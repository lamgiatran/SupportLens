"""Working document-retrieval slice. Missing model artifacts return explicit 503."""

import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request

from .contracts import AssistRequest, AssistResponse, BankingRequest, Latency
from .evaluation import Gate
from .retrieval import BM25Chunks, BM25Documents
from .request_logging import RequestLogMiddleware


def create_app(retriever=None, docs_gate: Gate | None = None,
               log_path: str | Path | None = None, *, log_questions: bool = False,
               banking=None) -> FastAPI:
    app = FastAPI(title="SupportLens", version="0.3.0")
    if log_path:
        app.add_middleware(RequestLogMiddleware, path=log_path, log_questions=log_questions)

    @app.get("/health")
    def health():
        return {"status": "ok", "docs_retrieval_ready": retriever is not None,
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
        if retriever is None or request.retriever != "R1":
            raise HTTPException(503, detail={"code": "retriever_not_loaded", "retriever": request.retriever})
        if docs_gate and (docs_gate.artifact_version != retriever.version or docs_gate.method != "R1"):
            raise HTTPException(503, detail={"code": "threshold_artifact_mismatch"})
        query_audit = {}
        passage_mode = isinstance(retriever, BM25Chunks)
        if passage_mode:
            query = retriever.prepare_query(request.question, [t.model_dump() for t in request.history])
            query_text = query.text
            query_audit = {"query_original_tokens": query.original_tokens, "query_used_tokens": query.used_tokens,
                           "query_truncated": query.truncated,
                           "current_question_truncated": query.current_question_truncated}
        else:
            query_text = request.question
        evidence = retriever.search(query_text, k=5)
        elapsed = (perf_counter() - start) * 1000
        score = evidence[0].score if evidence else None
        reject = not evidence or (docs_gate is not None and not docs_gate.accepts(score))
        response = AssistResponse(
            request_id=request_id, mode="docs_qa", model_scope="multidoc2dial",
            artifact_version=retriever.version,
            decision="handoff" if reject else "retrieved",
            reason=("retrieval_below_dev_threshold" if reject else "retrieval_only_generator_not_loaded")
                   + (";history_not_used_current_question_only" if request.history and not passage_mode else ""),
            evidence=evidence, score=score, score_method="R1_chunk_bm25" if passage_mode else "R1_document_bm25",
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
    gate = Gate(**json.loads(Path(gate_path).read_text(encoding="utf-8"))["gate"]) if gate_path else None
    return create_app(retriever, docs_gate=gate,
                      log_path=os.environ.get("SUPPORTLENS_LOG", "runs/api_requests.jsonl"),
                      log_questions=os.environ.get("SUPPORTLENS_LOG_QUESTIONS", "0") == "1")


app = app_from_env()
