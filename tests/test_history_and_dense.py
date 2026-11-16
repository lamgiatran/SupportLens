import importlib.util
from pathlib import Path

import numpy as np
import pytest

from supportlens.docs_eval import binary_auroc, paired_dialogue_bootstrap
from supportlens.docs_gate import DocsGate, select_coverage_gate
from supportlens.retrieval import BM25Chunks, words
from supportlens.text import Chunk, build_query
from supportlens.dense import (DenseEncoder, DenseEncoderConfig, DenseChunks, HybridChunks,
                               build_corpus_cache, load_corpus_cache, corpus_identity)
from supportlens.api import create_app
from supportlens.evaluation import Gate
from supportlens.generation import Prompt, make_prompt
from supportlens.docs_eval import visible_gold_hit
from fastapi.testclient import TestClient


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.asarray([[1., 0.] if "alpha" in t else [0., 1.] for t in texts], dtype=np.float32)


def dense_fixture(tokenizer):
    chunks = [Chunk(str(i), "d", str(i), "", text, 0, len(text), 4) for i, text in enumerate(
        ["alpha first", "beta second", "gamma third", "delta fourth"]) ]
    base = BM25Chunks(chunks, tokenizer, history_turns=2)
    backend = FakeBackend()
    config = DenseEncoderConfig("test/encoder", "a" * 40, dimension=2)
    encoder = DenseEncoder(config, backend=backend)
    matrix = np.asarray([[1., 0.], [0., 1.], [0.6, 0.8], [-1., 0.]], dtype=np.float32)
    dense = DenseChunks(base, encoder, matrix)
    return base, dense, HybridChunks(base, dense), backend


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_last_two_messages_are_unambiguous_and_current_query_is_first(tokenizer):
    history = [{"content": "far past"}, {"content": "older user"}, {"content": "recent assistant"}]
    assert build_query("current question", history, tokenizer, history_turns=2).text == "current question\nrecent assistant\nolder user"
    assert build_query("current question", history, tokenizer, history_turns=0).text == "current question"
    with pytest.raises(ValueError):
        build_query("current", history, tokenizer, history_turns=-1)


def test_sparse_bm25_scores_are_exact_including_repeats_oov_and_negative_idf(tokenizer):
    chunks = [Chunk(str(i), "d", str(i), "", t, 0, len(t), 5) for i, t in enumerate(
        ["common common alpha", "common beta", "common gamma", "common delta"]) ]
    retriever = BM25Chunks(chunks, tokenizer)
    for query in ["common alpha common", "absent", "", "beta gamma"]:
        np.testing.assert_array_equal(retriever.scores(query), retriever.index.get_scores(words(query)))
    short = BM25Chunks(chunks, tokenizer, history_turns=2)
    assert short.version != retriever.version


def test_coverage_gate_accepts_whole_ties_and_does_not_promise_low_risk():
    gate = select_coverage_gate([3, 2, 2, 1], [True] * 4, method="R1", artifact_version="v",
                                split="dev", score_name="bm25_top1_per_term_v1", target_coverage=0.5)
    assert gate.threshold == 1.5 and sum(gate.accepts(s) for s in [3, 2, 2, 1]) == 3
    with pytest.raises(ValueError, match="held-out"):
        select_coverage_gate([1], [False], method="R1", artifact_version="v", split="heldout", score_name="s")


def test_auroc_ties_and_direction():
    assert binary_auroc([1, 1], [True, False]) == 0.5
    assert binary_auroc([1, 0], [True, False]) == 1
    assert binary_auroc([0, 1], [True, False]) == 0
    assert binary_auroc([1], [True]) is None
    with pytest.raises(ValueError):
        binary_auroc([float("nan"), 1], [True, False])


def test_paired_bootstrap_resamples_dialogues_not_independent_turns():
    result = paired_dialogue_bootstrap(["long"] * 9 + ["short"], [1] * 9 + [0], [0] * 10,
                                       repeats=500, seed=42)
    assert result["delta_left_minus_right"] == 0.9
    assert result["ci95"] == [0.0, 1.0] and result["dialogues"] == 2


def test_analysis_entry_points_reject_heldout_before_touching_data():
    with pytest.raises(ValueError, match="dev-only"):
        script("ablate_history").run(None, None, None, None, split="heldout")
    with pytest.raises(ValueError, match="dev"):
        script("gate_scores").analyze([{"split": "heldout", "gold_ids": ["x"]}])
    with pytest.raises(ValueError, match="dev-only"):
        script("eval_docs_retrievers").run(None, None, None, None, None, split="heldout")


def test_dense_cosine_rankings_and_hybrid_confidence_are_separate_from_rrf(tokenizer):
    base, dense, hybrid, _ = dense_fixture(tokenizer)
    vector = np.array([1., 0.], dtype=np.float32)
    assert [e.evidence_id for e in dense.search_encoded(vector)] == ["0", "2", "1", "3"]
    result = hybrid.search_encoded("alpha", vector)
    assert hybrid.confidence_encoded(result, vector) == max(
        float(dense.matrix[dense.by_id[e.evidence_id]] @ vector) for e in result[:3])
    assert max(e.score for e in result) < 0.1  # Fusion scale is never the gate scale.
    with pytest.raises(ValueError, match="identical"):
        HybridChunks(BM25Chunks(base.chunks, tokenizer, history_turns=1), dense)


def test_corpus_cache_reuses_history_changes_but_rejects_encoder_or_row_changes(tmp_path, tokenizer):
    base, dense, _, _ = dense_fixture(tokenizer)
    build_corpus_cache(base, dense.encoder, tmp_path)
    other = BM25Chunks(base.chunks, tokenizer, history_turns=0)
    matrix, manifest = load_corpus_cache(other, dense.encoder, tmp_path)
    assert matrix.shape == (4, 2) and not matrix.flags.writeable
    assert corpus_identity(base, dense.encoder.config) == corpus_identity(other, dense.encoder.config)
    wrong = DenseEncoder(DenseEncoderConfig("test/encoder", "b" * 40, dimension=2), backend=FakeBackend())
    with pytest.raises(ValueError, match="identity"):
        load_corpus_cache(base, wrong, tmp_path)
    with (tmp_path / "corpus.npz").open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="bytes"):
        load_corpus_cache(base, dense.encoder, tmp_path)


def test_api_warms_shared_dense_before_requests_and_uses_offline_scores(tokenizer):
    base, dense, hybrid, backend = dense_fixture(tokenizer)
    registry = {"R1": base, "R2": dense, "R3": hybrid}
    gates = {m: DocsGate(m, obj.version, -100, "dev", obj.score_name, 3, "coverage", 0.7)
             for m, obj in registry.items()}
    with TestClient(create_app(retrievers=registry, docs_gates=gates)) as client:
        assert backend.calls == 1 and dense.encoder.ready
        assert client.get("/health").json()["docs_encoder_ready"] is True
        for method, obj in registry.items():
            evidence, score = obj.retrieve("alpha")
            result = client.post("/v1/assist", json={"mode": "docs_qa", "question": "alpha", "retriever": method}).json()
            assert result["score"] == pytest.approx(score)
            assert result["score_method"] == obj.score_name and result["decision"] == "retrieved"
            assert [e["evidence_id"] for e in result["evidence"]] == [e.evidence_id for e in evidence]
        assert len(client.get("/health").json()["cold_encoder_warmup_ms"]) == 1


def test_old_raw_threshold_cannot_be_silently_applied_to_normalized_confidence(tokenizer):
    base, _, _, _ = dense_fixture(tokenizer)
    old = Gate("R1", base.version, 1.0, "dev")
    response = TestClient(create_app(base, old)).post("/v1/assist", json={"mode": "docs_qa", "question": "alpha"})
    assert response.status_code == 503 and response.json()["detail"]["code"] == "threshold_score_policy_mismatch"


def test_gate_fit_binds_method_score_and_uses_top3_instead_of_top5():
    rows = [{"split": "dev", "query_id": "a", "method": "R1", "artifact_version": "v",
             "score_name": "bm25_top1_per_term_v1", "gate_score": 1.,
             "retrieved": ["a", "b", "c", "gold"], "gold_ids": ["gold"]}]
    fit = script("fit_docs_gate").fit
    report = fit(rows, method="R1")
    assert report["operating_point"]["risk"] == 1.0 and report["gate"]["target_k"] == 3
    with pytest.raises(ValueError, match="score policy"):
        fit([rows[0] | {"score_name": "raw_top1"}], method="R1")
    with pytest.raises(ValueError, match="dev-only"):
        fit(rows + [rows[0] | {"split": "heldout"}], method="R1")
    with pytest.raises(ValueError, match="smoke subsets"):
        fit(rows, method="R1", expected_qrels={"a": {"strict": ["gold"]}, "missing": {"strict": ["gold"]}})


def test_prompt_support_audit_excludes_title_and_trimmed_evidence(tokenizer):
    chunk = Chunk("c", "d", "doc", "alpha", "alpha\nabcdefghij", 0, 10, 12)
    raw = {"d": {"doc": {"doc_text": "abcdefghij", "spans": {"s": {"start_sp": 6, "end_sp": 10}}}}}
    refs = [{"doc_id": "doc", "id_sp": "s"}]
    # Visible title + first six body chars do not reach the gold span at all.
    prompt = Prompt("", 0, 0, visible_characters={"c": len("alpha\nabcdef")})
    assert not visible_gold_hit(prompt, {"c": chunk}, "d", refs, raw)
    prompt = Prompt("", 0, 0, visible_characters={"c": len("alpha\nabcdefgh")})
    assert visible_gold_hit(prompt, {"c": chunk}, "d", refs, raw)
    long_text = chunk.text + " alpha beta gamma" * 10
    packed = make_prompt("current question", [{"evidence_id": "c", "text": long_text}], tokenizer, evidence_budget=6)
    assert 0 < packed.visible_characters["c"] < len(long_text)


def test_dense_initialization_lock_loads_model_once_for_concurrent_callers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace
    import sys
    import time
    loads, barrier = [], Barrier(3)
    def factory(*args, **kwargs):
        loads.append(1)
        time.sleep(0.02)
        return FakeBackend()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory))
    encoder = DenseEncoder(DenseEncoderConfig("test/encoder", "a" * 40, dimension=2), local_files_only=True)
    def load():
        barrier.wait()
        return encoder._load()
    with ThreadPoolExecutor(max_workers=3) as pool:
        models = list(pool.map(lambda _: load(), range(3)))
    assert len(loads) == 1 and all(model is models[0] for model in models)
