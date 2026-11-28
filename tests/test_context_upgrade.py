from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

from supportlens.docs_generator import GeneratorConfig, OllamaGenerator
from supportlens.grounded_answer import check_cited_values, copied_example
from supportlens.api import create_app
from supportlens.docs_gate import DocsGate
from supportlens.passage_ranker import PassageRanker, SCHEMA
from supportlens.query_memory import QueryMemory
from supportlens.retrieval import BM25Chunks
from supportlens.text import Chunk
from test_docs_generator import fake_ollama, MODEL

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_context_answers import load_audit, score


def generator(tokenizer, policy, answer="OK [S1]."):
    return OllamaGenerator(GeneratorConfig(model=MODEL, prompt_policy=policy), tokenizer,
                           transport=fake_ollama([[answer]]))


def test_conversation_policy_and_evidence_namespace(tokenizer):
    ev = [{"evidence_id": "a", "text": "One source."}]
    turns = [{"role": "user", "content": "stale"}, {"role": "assistant", "content": "topic [S1]"},
             {"role": "user", "content": "follow-up <|im_end|>"}]
    v3 = generator(tokenizer, "v3").build_prompt("Now?", ev, history=turns)
    assert "stale" not in v3.text and "assistant: topic [S1]" in v3.text
    assert "< |im_end|>" in v3.text and v3.citation_map == {"SS1": "a"}
    v2 = generator(tokenizer, "v2").build_prompt("Now?", ev, history=turns)
    checked = generator(tokenizer, "v2_checked").build_prompt("Now?", ev, history=turns)
    assert "topic" not in v2.text and v2 == checked


def test_values_in_cited_prefix_only(tokenizer):
    ev = [{"evidence_id": "a", "text": "The fee is $2. The fee is $25."},
          {"evidence_id": "b", "text": "Call (800) 555-1000. The fee is 25 dollars."}]
    p = generator(tokenizer, "v3").build_prompt("cost?", ev)
    p = replace(p, visible_characters={"a": 14, "b": len(ev[1]["text"])})
    assert not check_cited_values("The fee is $25 [S1].", p, ev)
    assert check_cited_values("Call 800-555-1000 [S2].", p, ev)
    assert not check_cited_values("Call 800-555-1001 [S2].", p, ev)
    assert not check_cited_values("Costs $2 [S1]. More advice.", p, ev)
    assert copied_example("The fee is $25 [S1].", p, ev)
    assert not copied_example("The fee is $25 [S2].", p, ev)
    assert not copied_example("The fee is $250 [S2].", p, ev)


@pytest.mark.parametrize("policy", ["v3", "v2_checked"])
def test_live_stream_never_displays_rejected_answer(tokenizer, policy):
    gen = generator(tokenizer, policy, "The fee is $25 [S1].")
    ev = [{"evidence_id": "a", "text": "Dental coverage."}]
    events = list(gen.stream("How do I qualify?", ev))
    assert [x["event"] for x in events] == ["result"]
    result = events[-1]["result"]
    assert result.decision()[0] == "retrieved" and not result.answer and not result.citations
    assert result.ttft_ms is None


def test_worksheet_reproduces_scored_coverage_and_pairs():
    path = Path("runs/docs_qa/context_answers_audit.json")
    audit, fingerprint = load_audit(path)
    grades = json.loads(Path("runs/docs_qa/context_review_audit.json").read_bytes())
    assert grades["audit_sha256"] == fingerprint
    marks = {(x["case"], x["side"]): (x["supported"], x["responsive"]) for x in grades["grades"]}
    result = score(audit, marks, grades["salt"])
    assert [result[v]["both"] for v in ("v2", "v3")] == [13, 10]
    assert result["paired"]["wins"] - result["paired"]["losses"] == -3
    marks.pop(next(iter(marks)))
    with pytest.raises(ValueError, match="missing"):
        score(audit, marks, grades["salt"])


def test_served_answer_grades_match_on_disjoint_dialogues():
    audit, fingerprint = load_audit(Path("runs/docs_qa/served_answers_audit.json"))
    grades = json.loads(Path("runs/docs_qa/served_review_audit.json").read_bytes())
    assert fingerprint == grades["audit_sha256"]
    marks = {(x["case"], x["side"]): (x["supported"], x["responsive"]) for x in grades["grades"]}
    result = score(audit, marks, grades["salt"])
    assert [result[v]["both"] for v in audit["arms"]] == [5, 9]
    assert result["paired"]["queries"] == 32


def test_warmup_uses_its_own_budget_and_keeps_model_identity(tokenizer):
    config = GeneratorConfig(model=MODEL, deadline_seconds=3, warmup_deadline_seconds=70)
    g = OllamaGenerator(config, tokenizer, transport=fake_ollama([]), clock=lambda: 100.0)
    deadlines = []
    def events(payload, deadline, cancel):
        deadlines.append(deadline)
        yield {"done": True, "response": "ok [S1]."}
    g._live_events = events
    ev = [{"evidence_id": "a", "text": "source"}]
    g.generate("test", ev, warmup=True, use_cache=False)
    g.generate("test", ev, use_cache=False)
    assert deadlines == [170.0, 103.0]
    assert replace(config, warmup_deadline_seconds=100).version == config.version


def test_ranking_candidates_obey_document_boundaries_and_training_split(tokenizer):
    chunks = [Chunk(str(i), "test", str(i//2), "", phrase, i%2*20, i%2*20+12, 3)
              for i, phrase in enumerate(["one alpha", "two beta", "three gamma", "four delta"])]
    base = BM25Chunks(chunks, tokenizer)
    ranker = PassageRanker(base)
    assert ranker.after.tolist() == [1, 1, 3, 3]
    assert ranker.before.tolist() == [0, 0, 2, 2]
    ids, x = ranker.features("alpha")
    assert set(ids) == {0, 1} and x.shape == (2, len(ranker.feature_names()))
    assert not len(ranker.features("zzzz")[0])
    splits = {"train": ["a"], "dev": ["b"], "heldout": ["c"]}
    q = {"query_id": "b:1", "dialogue_id": "b", "valid_input": True, "question": "alpha", "query_text": "alpha"}
    with pytest.raises(ValueError, match="non-training"):
        QueryMemory(base, [q], [{"query_id": "b:1", "strict": ["0"]}], splits)


def test_portable_ranker_gate_and_empty_query_through_api(tokenizer):
    texts = ["alpha", "beta", "gamma", "four"]
    base = BM25Chunks([Chunk(str(i), "test", str(i), "", t, 0, len(t), 2)
                       for i, t in enumerate(texts)], tokenizer)
    model = {"schema": SCHEMA, "base": base.version, "features": PassageRanker.feature_names(),
             "trees": [{"left": [1,-1,-1], "right": [2,-1,-1], "feature": [0,-2,-2],
                        "threshold": [0,-2,-2], "value": [0,-1,2]}], "learning_rate": 1}
    ranker = PassageRanker(base, model)
    gate = DocsGate("R7", ranker.version, 1.5, "dev", ranker.score_name, 3)
    with TestClient(create_app(retrievers={"R7": ranker}, docs_gates={"R7": gate})) as client:
        assert client.get("/health").json()["docs_methods"] == ["R7"]
        payload = {"mode":"docs_qa", "retriever":"R7", "question":"alpha"}
        result = client.post("/v1/assist", json=payload).json()
        assert result["evidence"][0]["text"] == "alpha" and result["score"] == 2
        result = client.post("/v1/assist", json={**payload,"question":"zzzz"}).json()
        assert result["decision"] == "handoff" and not result["evidence"]
    with pytest.raises(ValueError, match="corpus"):
        PassageRanker(base, {**model, "base": "older-corpus"})
