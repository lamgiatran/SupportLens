"""Docs-QA generation against a fake Ollama server: no model, no network."""

import asyncio
import json
from threading import Event
from time import monotonic

import httpx
import pytest
from fastapi.testclient import TestClient

from supportlens.api import create_app
from supportlens.docs_generator import GeneratorConfig, OllamaGenerator, classify, generator_from_env
from supportlens.retrieval import BM25Documents


MODEL = "qwen2.5:1.5b"


def fake_ollama(answers, *, models=(MODEL,), calls=None, status=200):
    """`answers` is a list of token lists; each /api/generate call consumes one (warm-up included)."""
    queue = list(answers)

    def handler(request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        if request.url.path == "/api/generate":
            body = json.loads(request.content)
            if calls is not None:
                calls.append(body)
            tokens = queue.pop(0) if queue else ["ok"]
            if status != 200:
                return httpx.Response(status, json={"error": "boom"})
            lines = [json.dumps({"response": t, "done": False}) for t in tokens]
            lines.append(json.dumps({"response": "", "done": True, "eval_count": len(tokens)}))
            return httpx.Response(200, content=("\n".join(lines) + "\n").encode())
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def make_generator(tokenizer, answers, **kwargs):
    config = GeneratorConfig(model=MODEL, num_ctx=512, num_predict=32, evidence_budget_tokens=200,
                             **kwargs.pop("config", {}))
    return OllamaGenerator(config, tokenizer, transport=fake_ollama(answers, **kwargs))


def evidence(*pairs):
    return [{"evidence_id": i, "text": t} for i, t in pairs]


def test_cited_answer_is_accepted_and_maps_to_real_ids(tokenizer):
    gen = make_generator(tokenizer, [["Use the ", "alpha form [S1]."]])
    result = gen.generate("current question", evidence(("doc-a", "alpha"), ("doc-b", "beta")))
    assert result.status == "answer" and result.citations == ["doc-a"]
    assert result.decision() == ("answer", "generated_with_verified_citations")
    assert result.ttft_ms is not None and result.cache_status == "miss"


@pytest.mark.parametrize("text, status, decision", [
    ("The answer is alpha.", "no_citation", "retrieved"),
    ("Alpha [S9].", "invalid_citation", "retrieved"),
    ("I cannot answer from these sources; please request human review.", "refused", "handoff"),
    ("   ", "empty", "retrieved"),
])
def test_unverifiable_answers_never_publish(tokenizer, text, status, decision):
    gen = make_generator(tokenizer, [[text]])
    result = gen.generate("current question", evidence(("doc-a", "alpha")))
    assert result.status == status and result.decision()[0] == decision
    if status != "answer":
        assert result.citations == []


def test_cache_hit_skips_the_model_and_only_caches_valid_outcomes(tokenizer):
    calls = []
    gen = make_generator(tokenizer, [["Alpha [S1]."], ["no cite"], ["no cite"]], calls=calls)
    first = gen.generate("current question", evidence(("doc-a", "alpha")))
    second = gen.generate("current question", evidence(("doc-a", "alpha")))
    assert (first.cache_status, second.cache_status) == ("miss", "hit")
    assert second.answer == first.answer and second.citations == ["doc-a"] and len(calls) == 1
    gen.generate("first", evidence(("doc-b", "beta")))
    gen.generate("first", evidence(("doc-b", "beta")))
    assert len(calls) == 3  # a no-citation answer is retried, not frozen in the cache


def test_deadline_cuts_a_slow_stream_and_discards_partial_text(tokenizer):
    ticks = iter(range(0, 1000, 10))  # each clock read advances 10 seconds
    config = GeneratorConfig(model=MODEL, num_ctx=512, num_predict=32, evidence_budget_tokens=200,
                             deadline_seconds=15)
    gen = OllamaGenerator(config, tokenizer, transport=fake_ollama([["a", "b", "c", "d"]]),
                          clock=lambda: next(ticks))
    result = gen.generate("current question", evidence(("doc-a", "alpha")))
    assert result.status == "timeout" and result.answer == "" and result.citations == []
    assert result.decision() == ("retrieved", "generator_timeout_fallback_to_evidence")


def test_cancel_stops_the_stream(tokenizer):
    gen = make_generator(tokenizer, [["a", "b", "c"]])
    cancel = Event()
    events = []
    for event in gen.stream("current question", evidence(("doc-a", "alpha")), cancel=cancel):
        events.append(event)
        cancel.set()
    assert events[-1]["result"].status == "cancelled"
    assert sum(e["event"] == "delta" for e in events) == 1


def test_server_error_falls_back(tokenizer):
    gen = OllamaGenerator(GeneratorConfig(model=MODEL, num_ctx=512, num_predict=32, evidence_budget_tokens=200),
                          tokenizer, transport=fake_ollama([], status=500))
    result = gen.generate("current question", evidence(("doc-a", "alpha")))
    assert result.status == "error" and result.error_type == "HTTPStatusError"


def test_journal_replay_requires_model_history_and_real_source_ids(tokenizer, tmp_path):
    calls = []
    config = GeneratorConfig(model=MODEL, num_ctx=512, num_predict=32, evidence_budget_tokens=200)
    def instance(digest):
        transport = fake_ollama([["Alpha [S1]."]] * 6, calls=calls)
        gen = OllamaGenerator(config, tokenizer, transport=transport, journal_dir=tmp_path)
        gen.model_digest = digest
        return gen
    ev = evidence(("id-a", "same source text"))
    gen = instance("digest-a")
    assert gen.generate("Where?", ev).cache_status == "miss"
    assert gen.generate("Where?", ev).cache_status == "hit"
    restarted = instance("digest-a")
    result = restarted.generate("Where?", ev)
    assert result.cache_status == "replay" and result.ttft_ms is None
    assert result.citations == ["id-a"] and len(calls) == 1
    moved = restarted.generate("Where?", evidence(("id-b", "same source text")))
    assert moved.cache_status == "miss" and moved.citations == ["id-b"]
    assert restarted.generate("Where?", ev, history=[{"role": "user", "text": "other"}]).cache_status == "miss"
    assert restarted.generate("Where?", ev, retrieval_version="different").cache_status == "miss"
    assert instance("digest-b").generate("Where?", ev).cache_status == "miss"
    assert len(calls) == len(list(tmp_path.rglob("*.jsonl"))) == 5
    for file in tmp_path.rglob("*.jsonl"):
        row = json.loads(file.read_text(encoding="utf-8"))
        assert "text" not in row["prompt"] and "input" not in row and "question" not in row["identity"]


def test_wall_deadline_cancels_a_stalled_read_and_journals_partial_only(tokenizer, tmp_path):
    closed = Event()
    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"response":"partial","done":false}\n'
            await asyncio.sleep(10)
        async def aclose(self):
            closed.set()
    config = GeneratorConfig(model=MODEL, deadline_seconds=0.4)
    gen = OllamaGenerator(config, tokenizer, journal_dir=tmp_path,
                          transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stalled())))
    before = monotonic()
    result = gen.generate("Where?", evidence(("p", "alpha")))
    assert monotonic() - before < 1.5 and closed.is_set()
    assert result.status == "timeout" and result.answer == "" and not result.citations
    row = json.loads(next(tmp_path.rglob("*.timeout.jsonl")).read_text(encoding="utf-8"))
    assert row["raw_text"] == "partial" and row["result"]["answer"] == ""
    assert gen.journal.replay(row["key"]) is None


def test_abandoned_stream_preserves_partial_without_replay(tokenizer, tmp_path):
    gen = OllamaGenerator(GeneratorConfig(model=MODEL), tokenizer, journal_dir=tmp_path,
                          transport=fake_ollama([["first ", "second [S1]."]]))
    stream = gen.stream("Where?", evidence(("p", "alpha")))
    assert next(stream)["text"] == "first "
    stream.close()
    row = json.loads(next(tmp_path.rglob("*.cancelled.jsonl")).read_text(encoding="utf-8"))
    assert row["raw_text"] == "first " and row["result"]["answer"] == ""
    assert gen.journal.replay(row["key"]) is None


def test_startup_requires_pulled_model_and_never_pulls(tokenizer):
    calls = []
    gen = make_generator(tokenizer, [], models=("other:7b",), calls=calls)
    gen.start()
    assert gen.state == "failed" and gen.error_type == "LookupError" and calls == []
    ok = make_generator(tokenizer, [["warm [S1]"]])
    ok.start()
    assert ok.ready and ok.warmup_ms is not None


def test_config_reads_default_generator_block_and_versions_prompt_policy(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(json.dumps({"generator": {"model": "m:1b", "num_ctx": 1024, "num_predict": 64,
                                              "stream": True, "ttft_target_ms": 3000}}))
    config = GeneratorConfig.load(path, deadline_seconds=12.0, base_url=None)
    assert (config.model, config.deadline_seconds, config.base_url) == ("m:1b", 12.0, "http://127.0.0.1:11434")
    same = GeneratorConfig.load(path, deadline_seconds=99.0)
    assert config.version == same.version  # deadline is serving policy, not answer identity
    assert GeneratorConfig.load(path, model="m:3b").version != config.version


def test_env_is_opt_in():
    assert generator_from_env({}) is None
    with pytest.raises(ValueError):
        generator_from_env({"SUPPORTLENS_GENERATOR": "openai"})


def test_classify_rejects_mixed_valid_and_invented_citations(tokenizer):
    gen = make_generator(tokenizer, [])
    prompt = gen.build_prompt("current question", evidence(("doc-a", "alpha")))
    assert classify("alpha [S1] and [S2]", prompt)[0] == "invalid_citation"


# ---------- API ----------

@pytest.fixture
def docs_rows():
    return [
        {"domain": "synthetic", "doc_id": "a", "title": "Address", "text": "alpha address change procedure"},
        {"domain": "synthetic", "doc_id": "b", "title": "Cars", "text": "beta vehicle registration"},
    ]


def api(tokenizer, docs_rows, answers, **kwargs):
    gen = make_generator(tokenizer, [["warm"]] + answers, **kwargs)
    return TestClient(create_app(BM25Documents(docs_rows), generator=gen)), gen


def test_api_returns_cited_answer_and_health_reports_generator(tokenizer, docs_rows):
    client, gen = api(tokenizer, docs_rows, [["Change it with the alpha form [S1]."]])
    with client:
        health = client.get("/health").json()
        assert health["docs_generator_ready"] is True and health["docs_generator"]["model"] == MODEL
        body = client.post("/v1/assist", json={"mode": "docs_qa", "question": "alpha address"}).json()
    assert body["decision"] == "answer" and body["answer"].endswith("[S1].")
    assert body["citations"] == [body["evidence"][0]["evidence_id"]]
    assert body["artifact_version"].endswith("+" + gen.version)
    assert body["latency"]["generate_ms"] is not None and body["cache_status"] == "miss"


def test_api_falls_back_to_evidence_on_uncited_answer(tokenizer, docs_rows):
    client, _ = api(tokenizer, docs_rows, [["It is alpha."]])
    with client:
        body = client.post("/v1/assist", json={"mode": "docs_qa", "question": "alpha address"}).json()
    assert body["decision"] == "retrieved" and body["answer"] is None and body["citations"] == []
    assert body["reason"].startswith("generator_no_citation_fallback_to_evidence")
    assert body["evidence"]


def test_api_without_ready_generator_keeps_retrieval_and_says_why(tokenizer, docs_rows):
    client, gen = api(tokenizer, docs_rows, [], models=("other:7b",))
    with client:
        assert client.get("/health").json()["docs_generator"]["state"] == "failed"
        body = client.post("/v1/assist", json={"mode": "docs_qa", "question": "alpha"}).json()
    assert body["decision"] == "retrieved" and body["reason"].startswith("retrieval_only_generator_not_ready")


def read_events(response):
    return [json.loads(line) for line in response.iter_lines() if line]


def test_stream_emits_start_deltas_done_in_order(tokenizer, docs_rows):
    client, _ = api(tokenizer, docs_rows, [["Use ", "alpha ", "[S1]."]])
    with client, client.stream("POST", "/v1/assist/stream",
                               json={"mode": "docs_qa", "question": "alpha address"}) as response:
        assert response.headers["content-type"].startswith("application/x-ndjson")
        events = read_events(response)
    kinds = [e["event"] for e in events]
    assert kinds == ["start", "delta", "delta", "delta", "done"]
    assert events[0]["generating"] is True and events[0]["evidence"]
    assert "".join(e["text"] for e in events if e["event"] == "delta") == events[-1]["response"]["answer"]
    assert events[-1]["response"]["decision"] == "answer"
    assert events[0]["request_id"] == events[-1]["response"]["request_id"]


def test_stream_done_overrides_partial_text_when_answer_is_rejected(tokenizer, docs_rows):
    client, _ = api(tokenizer, docs_rows, [["Made ", "up [S7]."]])
    with client, client.stream("POST", "/v1/assist/stream",
                               json={"mode": "docs_qa", "question": "alpha address"}) as response:
        events = read_events(response)
    done = events[-1]["response"]
    assert done["decision"] == "retrieved" and done["answer"] is None
    assert done["reason"].startswith("generator_invalid_citation")


def test_stream_banking_and_missing_retriever_use_same_errors(tokenizer, docs_rows):
    client, _ = api(tokenizer, docs_rows, [])
    with client:
        assert client.post("/v1/assist/stream", json={"mode": "banking_triage", "question": "card"}).status_code == 503
        missing = client.post("/v1/assist/stream", json={"mode": "docs_qa", "question": "a", "retriever": "R2"})
    assert missing.status_code == 503 and missing.json()["detail"]["code"] == "retriever_not_loaded"


def test_stream_without_generator_still_returns_evidence(docs_rows):
    client = TestClient(create_app(BM25Documents(docs_rows)))
    with client, client.stream("POST", "/v1/assist/stream", json={"mode": "docs_qa", "question": "beta"}) as r:
        events = read_events(r)
    assert [e["event"] for e in events] == ["start", "done"]
    assert events[0]["generating"] is False
    assert events[-1]["response"]["reason"].startswith("retrieval_only_generator_not_loaded")


def test_v2_prompt_asks_for_sentence_citations_and_v1_is_unchanged(tokenizer):
    v2 = make_generator(tokenizer, [])
    v1 = make_generator(tokenizer, [], config={"prompt_policy": "v1"})
    ev = evidence(("doc-a", "alpha"))
    p1, p2 = v1.build_prompt("current question", ev), v2.build_prompt("current question", ev)
    assert "Answer only using the supplied evidence" in p1.text and "{p}" not in p2.text
    assert "End every sentence" in p2.text and p2.text.index("Question:") < p2.text.rindex("[S1]")
    assert p1.citation_map == p2.citation_map == {"S1": "doc-a"}
    assert v1.version != v2.version
    with pytest.raises(ValueError):
        GeneratorConfig(prompt_policy="v9")
