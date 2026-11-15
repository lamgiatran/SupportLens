import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError
from supportlens.api import create_app
from supportlens.contracts import AssistRequest
from supportlens.evaluation import Gate
from supportlens.retrieval import BM25Documents, rrf


@pytest.mark.parametrize("payload", [
    {"mode": "banking_triage", "question": "hi", "model_scope": "course77"},
    {"mode": "docs_qa", "question": "hi", "gold_doc_id": "hidden-label"},
    {"mode": "docs_qa", "question": "   "},
])
def test_request_rejects_scope_mix_and_gold_inputs(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(AssistRequest).validate_python(payload)


def test_missing_banking_model_returns_503():
    response = TestClient(create_app()).post("/v1/assist", json={"mode": "banking_triage", "question": "card?"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "handoff50_artifact_missing"


def test_document_api_returns_real_evidence_without_claiming_generation(rows):
    response = TestClient(create_app(BM25Documents(rows))).post(
        "/v1/assist", json={"mode": "docs_qa", "question": "uniqueaddress"})
    assert response.status_code == 200
    result = response.json()
    assert result["evidence"][0]["doc_id"] == "a"
    assert result["decision"] == "retrieved" and result["answer"] is None
    assert result["evidence"][0]["unit"] == "document"


def test_gate_cannot_be_reused_for_another_artifact(rows):
    client = TestClient(create_app(BM25Documents(rows), Gate("R1", "wrong-version", 0.5, "dev")))
    assert client.post("/v1/assist", json={"mode": "docs_qa", "question": "uniqueaddress"}).status_code == 503


def test_fitted_deny_all_gate_hands_off(rows):
    retriever = BM25Documents(rows)
    client = TestClient(create_app(retriever, Gate("R1", retriever.version, None, "dev")))
    assert client.post("/v1/assist", json={"mode": "docs_qa", "question": "uniqueaddress"}).json()["decision"] == "handoff"


def test_rrf_can_compare_frozen_and_finetuned_with_same_bm25():
    assert rrf([["a", "b"], ["b", "a"]])[0][1] == pytest.approx(1/61 + 1/62)
