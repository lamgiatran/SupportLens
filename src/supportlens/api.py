"""Working document-retrieval slice. Missing model artifacts return explicit 503."""

import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .contracts import AssistRequest, AssistResponse, BankingRequest, Latency
from .evaluation import Gate
from .retrieval import BM25Documents


def create_app(retriever=None, docs_gate: Gate | None = None,
               log_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="SupportLens", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "docs_retrieval_ready": retriever is not None,
                "docs_generator_ready": False, "banking_model_ready": False}

    @app.post("/v1/assist", response_model=AssistResponse)
    def assist(request: AssistRequest):
        start = perf_counter()
        if isinstance(request, BankingRequest):
            raise HTTPException(503, detail={"code": "handoff50_artifact_missing",
                                            "message": "Train and load the separate 50-label model first."})
        if retriever is None or request.retriever != "R1":
            raise HTTPException(503, detail={"code": "retriever_not_loaded", "retriever": request.retriever})
        if docs_gate and (docs_gate.artifact_version != retriever.version or docs_gate.method != "R1"):
            raise HTTPException(503, detail={"code": "threshold_artifact_mismatch"})
        # Day 2 is DOCUMENT-level BM25 on current question only, clearly labelled.
        # Token-budgeted shared query + chunk retrieval is the day-3 issue.
        evidence = retriever.search(request.question, k=5)
        elapsed = (perf_counter() - start) * 1000
        score = evidence[0].score if evidence else None
        reject = not evidence or (docs_gate is not None and not docs_gate.accepts(score))
        response = AssistResponse(
            request_id=str(uuid4()), mode="docs_qa", model_scope="multidoc2dial",
            artifact_version=retriever.version,
            decision="handoff" if reject else "retrieved",
            reason="retrieval_below_dev_threshold" if reject else "retrieval_only_generator_not_loaded",
            evidence=evidence, score=score, score_method="R1_document_bm25",
            latency=Latency(total_ms=elapsed, retrieve_ms=elapsed),
        )
        if log_path:
            target = Path(log_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # No raw question/history in operational logs by default.
            record = response.model_dump(exclude={"evidence", "answer"})
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return response

    return app


def app_from_env():
    corpus = os.environ.get("SUPPORTLENS_CORPUS")
    retriever = BM25Documents.from_jsonl(corpus) if corpus else None
    gate_path = os.environ.get("SUPPORTLENS_DOCS_GATE")
    gate = Gate(**json.loads(Path(gate_path).read_text(encoding="utf-8"))["gate"]) if gate_path else None
    return create_app(retriever, docs_gate=gate, log_path=os.environ.get("SUPPORTLENS_LOG"))


app = app_from_env()
