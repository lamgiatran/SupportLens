import json

import httpx
import pytest
from fastapi.testclient import TestClient

from supportlens.api import create_app
from supportlens.generation import make_prompt, resolve_citations
from supportlens.retrieval import BM25Documents, words
from supportlens.text import build_query
from supportlens.ui_client import parse_assist_response


def test_history_formatter_adds_no_role_words_or_special_separator(tokenizer):
    query = build_query("current question", [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "recent answer"}], tokenizer)
    assert query.text == "current question\nrecent answer\nold"
    assert not {"user", "assistant", "sep"} & set(words(query.text))


@pytest.mark.parametrize("status,body", [(503, b"<html>Unavailable</html>"),
    (422, b"[]"), (200, b"not json"), (200, b'{"decision":"retrieved"}')])
def test_ui_handles_error_pages_and_invalid_success_contracts(status, body):
    value, error = parse_assist_response(httpx.Response(status, content=body))
    assert value is None and f"HTTP {status}" in error


def test_ui_accepts_the_actual_api_response(rows):
    response = TestClient(create_app(BM25Documents(rows))).post(
        "/v1/assist", json={"mode": "docs_qa", "question": "uniqueaddress"})
    value, error = parse_assist_response(response)
    assert error is None and value.evidence[0].doc_id == "a"


@pytest.mark.parametrize("log_questions", [False, True])
def test_log_covers_success_validation_and_unavailable_without_duplicates(tmp_path, rows, log_questions):
    path = tmp_path / "http.jsonl"
    client = TestClient(create_app(BM25Documents(rows), log_path=path, log_questions=log_questions))
    payloads = [
        {"mode": "docs_qa", "question": "uniqueaddress"},
        {"mode": "docs_qa", "question": " "},
        {"mode": "banking_triage", "question": "card?"}]
    responses = [client.post("/v1/assist", json=p) for p in payloads]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["status_code"] for r in records] == [200, 422, 503]
    assert len({r["request_id"] for r in records}) == 3
    for record, response, payload in zip(records, responses, payloads):
        assert record["request_id"] == response.headers["x-request-id"]
        assert ("question" in record) is log_questions
        if log_questions:
            assert record["question"] == payload["question"]
    assert responses[0].json()["request_id"] == records[0]["request_id"]


def test_log_covers_unhandled_server_error_and_preserves_500(tmp_path):
    class BrokenRetriever:
        def search(self, *args, **kwargs):
            raise RuntimeError("synthetic failure")
    path = tmp_path / "http.jsonl"
    client = TestClient(create_app(BrokenRetriever(), log_path=path), raise_server_exceptions=False)
    assert client.post("/v1/assist", json={"mode": "docs_qa", "question": "hello"}).status_code == 500
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status_code"] == 500 and record["error_type"] == "RuntimeError"


def test_history_not_used_is_explicit_and_does_not_change_day_two_retrieval(rows):
    client = TestClient(create_app(BM25Documents(rows)))
    result = client.post("/v1/assist", json={"mode": "docs_qa", "question": "uniqueaddress",
        "history": [{"role": "user", "content": "vehicle registration"}]}).json()
    assert "history_not_used_current_question_only" in result["reason"]
    assert result["evidence"][0]["doc_id"] == "a"


def test_citation_aliases_hide_hashes_and_map_back_to_real_ids(tokenizer):
    ids = ["0123456789abcdefabcd", "fedcba9876543210abcd"]
    prompt = make_prompt("current question", [
        {"evidence_id": key, "text": "alpha beta"} for key in ids], tokenizer)
    assert all(key not in prompt.text for key in ids)
    assert "[S1] alpha beta" in prompt.text and "[S2] alpha beta" in prompt.text
    citations, invalid = resolve_citations("alpha [S2], beta [S1], repeated [S2], bad [S9]", prompt)
    assert citations == [ids[1], ids[0]] and invalid == ["S9"]


def test_unpacked_evidence_cannot_be_cited(tokenizer):
    prompt = make_prompt("current question", [
        {"evidence_id": str(i), "text": "alpha " * 100} for i in range(4)],
        tokenizer, evidence_budget=12)
    assert set(prompt.citation_map) == {"S1"}
    assert resolve_citations("[S2]", prompt) == ([], ["S2"])
