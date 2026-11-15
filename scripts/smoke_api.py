"""A real-corpus ASGI smoke check; not a benchmark or a live-server p95 test."""

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supportlens.api import create_app
from supportlens.retrieval import BM25Documents


def main():
    retriever = BM25Documents.from_jsonl("data/processed/documents.jsonl")
    client = TestClient(create_app(retriever))
    response = client.post("/v1/assist", json={"mode": "docs_qa", "question": "How do I change my address?"})
    response.raise_for_status()
    value = response.json()
    report = {"check": "in_process_ASGI_smoke_not_benchmark", "http_status": response.status_code,
              "decision": value["decision"], "artifact_version": value["artifact_version"],
              "corpus_documents": len(retriever.rows),
              "evidence": [{k: e[k] for k in ("evidence_id", "doc_id", "domain", "unit")} for e in value["evidence"]],
              "generator_connected": False}
    Path("runs").mkdir(exist_ok=True)
    Path("runs/api_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
