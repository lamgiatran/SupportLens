from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess

from fastapi.testclient import TestClient
import pytest

from supportlens.api import create_app
from supportlens.data import QrelsMapper
from supportlens.docs_pipeline import build_docs_bundle, dialogue_split
from supportlens.generation import make_prompt, resolve_citations
from supportlens.retrieval import BM25Chunks, deduplicate_chunks
from supportlens.text import Chunk


def test_strict_relevance_rejects_one_character_touch_but_keeps_half_span():
    raw = {"d": {"x": {"doc_text": "abcdefghijklmnopqrst", "spans": {
        "1": {"start_sp": 5, "end_sp": 15, "text_sp": "fghijklmno"}}}}}
    chunks = [Chunk("touch", "d", "x", "", "abcdef", 0, 6, 1),
              Chunk("half", "d", "x", "", "klmnopqrst", 10, 20, 1)]
    result = QrelsMapper(raw, chunks).map("d", [{"doc_id": "x", "id_sp": "1"}])
    assert result["strict"] == ["half"] and result["lenient"] == ["half", "touch"]
    assert result["references"][0]["coverage"][0]["span_fraction"] == 0.1


def test_invalid_and_mismatched_spans_are_audited_instead_of_becoming_gold():
    raw = {"d": {"x": {"doc_text": "abc[2]def", "spans": {
        "1": {"start_sp": 0, "end_sp": 3, "text_sp": "wrong"},
        "2": {"start_sp": 2, "end_sp": 100}}}}}
    chunk = Chunk("c", "d", "x", "", "abc[2]def", 0, 9, 3)
    result = QrelsMapper(raw, [chunk]).map("d", [{"doc_id": "x", "id_sp": str(i)} for i in [1, 2, 3]])
    assert result["strict"] == [] and result["lenient"] == []
    assert {r["issue"] for r in result["issues"]} == {"span_text_mismatch", "invalid_offsets", "unknown_span"}
    assert raw["d"]["x"]["doc_text"] == "abc[2]def"


def test_cross_domain_span_and_repeated_reference_never_create_extra_positives():
    raw = {"d": {"x": {"doc_text": "abcdef", "spans": {"1": {"start_sp": 0, "end_sp": 6}}}}}
    chunks = [Chunk("right", "d", "x", "", "abcdef", 0, 6, 1),
              Chunk("other-domain", "other", "x", "", "abcdef", 0, 6, 1)]
    result = QrelsMapper(raw, chunks).map("d", [{"doc_id": "x", "id_sp": "1"}] * 2)
    assert result["strict"] == ["right"] and len(result["references"]) == 1


def test_heavy_overlap_is_removed_but_complementary_same_document_chunks_survive():
    chunks = [Chunk("a", "d", "x", "", "a", 0, 100, 1),
              Chunk("duplicate", "d", "x", "", "b", 10, 90, 1),
              Chunk("complement", "d", "x", "", "c", 90, 180, 1),
              Chunk("other", "d", "y", "", "d", 0, 100, 1)]
    result = deduplicate_chunks([(c, 4-i) for i, c in enumerate(chunks)], k=3)
    assert [c.chunk_id for c, _ in result] == ["a", "complement", "other"]


def test_prefixed_citation_lists_do_not_confuse_original_numeric_links(tokenizer):
    prompt = make_prompt("current question", [
        {"evidence_id": "hash-a", "text": "alpha [2] [4] [5 ]"},
        {"evidence_id": "hash-b", "text": "beta"}], tokenizer)
    assert "alpha [2] [4] [5 ]" in prompt.text
    assert resolve_citations("Links [4], [5 ], [2] and [1, 2]. Sources [S1, S2].", prompt) == (
        ["hash-a", "hash-b"], [])
    assert resolve_citations("Sources [S1, 2, S9]", prompt) == (["hash-a", "hash-b"], ["S9"])
    assert resolve_citations("Only a document link [2]", prompt) == ([], [])


def test_reserved_citation_namespace_is_changed_when_source_already_uses_it(tokenizer):
    prompt = make_prompt("current question", [{"evidence_id": "real", "text": "alpha [S1]"}], tokenizer)
    assert prompt.citation_prefix == "SS" and "alpha [S1]" in prompt.text
    assert resolve_citations("Literal [S1]. Citation [SS1].", prompt) == (["real"], [])


def test_dialogue_split_is_deterministic_disjoint_and_keeps_domains():
    train = {"d": [{"dial_id": "train"}]}
    val = {d: [{"dial_id": d+str(i)} for i in range(6)] for d in ["a", "b"]}
    result = dialogue_split(train, val)
    assert result == dialogue_split(train, val)
    ids = result["dialogues"]
    assert not set(ids["train"]) & (set(ids["dev"]) | set(ids["heldout"]))
    assert not set(ids["dev"]) & set(ids["heldout"])
    assert all(sum(x.startswith(d) for x in ids["dev"]) == 3 for d in val)
    with pytest.raises(ValueError, match="overlapping"):
        dialogue_split(train, {"d": [{"dial_id": "train"}, {"dial_id": "other"}]})


@pytest.mark.parametrize("config", [{"query_max_tokens": 512}, {"misspelled_budget": 220}])
def test_bad_query_budget_or_unknown_config_fails_before_reading_data(tmp_path, config):
    with pytest.raises(ValueError):
        build_docs_bundle(tmp_path / "missing", tmp_path / "no-tokenizer", tmp_path / "out", config=config)


def test_bundle_round_trip_and_api_use_the_same_passage_history_policy(tmp_path, tokenizer):
    raw = tmp_path / "raw"
    raw.mkdir()
    body = "alpha beta gamma " * 30
    docs = {"doc_data": {"d": {"x": {"title": "title", "doc_text": body,
        "spans": {"1": {"start_sp": 0, "end_sp": 5, "text_sp": "alpha"}}}}}}
    (raw / "multidoc2dial_doc.json").write_text(json.dumps(docs))
    def dialogue(key):
        return {"dial_id": key, "turns": [
            {"turn_id": 1, "role": "user", "utterance": "current question"},
            {"turn_id": 2, "role": "agent", "utterance": "SECRET_TARGET", "references": [{"doc_id": "x", "id_sp": "1"}]}]}
    for split, ids in [("train", ["t"]), ("validation", ["v1", "v2", "v3", "v4"])]:
        (raw / f"multidoc2dial_dial_{split}.json").write_text(json.dumps({"dial_data": {"d": [dialogue(i) for i in ids]}}))
    token_path = tmp_path / "tokenizer.json"
    tokenizer.backend.save(str(token_path))
    bundle = tmp_path / "bundle"
    report = build_docs_bundle(raw, token_path, bundle, config={"chunk_max_tokens": 12, "chunk_overlap": 2})
    retriever = BM25Chunks.from_bundle(bundle, token_path)
    assert report["max_chunk_tokens"] <= 12 and report["chunks"] > 1
    assert "SECRET_TARGET" not in (bundle / "queries.train.jsonl").read_text()
    history = [{"role": "user", "content": "old " * 500}]
    query = retriever.prepare_query("current question", history)
    direct = retriever.search(query.text)
    log = tmp_path / "requests.jsonl"
    response = TestClient(create_app(retriever, log_path=log)).post("/v1/assist", json={
        "mode": "docs_qa", "question": "current question", "history": history})
    assert response.status_code == 200
    value = response.json()
    assert [e["evidence_id"] for e in value["evidence"]] == [e.evidence_id for e in direct]
    assert all(e["unit"] == "chunk" for e in value["evidence"])
    assert "history_not_used" not in value["reason"]
    assert json.loads(log.read_text())["query_truncated"] is True
    with (bundle / "chunks.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="bytes"):
        BM25Chunks.from_bundle(bundle, token_path)


def test_hashed_json_assets_keep_bytes_after_windows_style_git_checkout(tmp_path):
    repo, exported = tmp_path / "source", tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / ".gitattributes", repo / ".gitattributes")
    payloads = {"splits.json": b'{"ids":["a","b"]}\n',
                "qrels_audit.jsonl": b'{"query_id":"a"}\n{"query_id":"b"}\n'}
    for name, data in payloads.items():
        (repo / name).write_bytes(data)
    subprocess.run(["git", "-c", "core.autocrlf=true", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=SupportLens Test", "-c", "user.email=test@example.invalid",
                    "commit", "--quiet", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    exported.mkdir()
    # Use Git's checkout path directly; no remote transport or MSYS shell needed.
    subprocess.run(["git", "-c", "core.autocrlf=true", "checkout-index", "--all",
                    "--prefix=" + exported.as_posix() + "/"], cwd=repo, check=True, capture_output=True)
    assert all((exported / name).read_bytes() == data for name, data in payloads.items())
