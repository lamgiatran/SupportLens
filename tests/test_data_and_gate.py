import json
import pytest
from supportlens.data import examples, map_references
from supportlens.evaluation import risk_coverage, select_gate
from supportlens.text import Chunk


def test_target_answer_and_future_turns_never_enter_query(tmp_path):
    turns = [{"turn_id": 1, "role": "user", "utterance": "CURRENT"},
             {"turn_id": 2, "role": "agent", "utterance": "SECRET_TARGET", "references": []},
             {"turn_id": 3, "role": "user", "utterance": "FUTURE"}]
    file = tmp_path / "raw.json"
    file.write_text(json.dumps({"dial_data": {"d": [{"dial_id": "x", "turns": turns}]}}))
    row = list(examples(file))[0]
    assert row["question"] == "CURRENT" and row["history"] == []
    assert "SECRET_TARGET" not in json.dumps(row)


def test_multi_reference_span_across_chunk_boundary():
    raw = {"d": {"x": {"doc_text": "abcdefghij", "spans": {
        "1": {"start_sp": 3, "end_sp": 7}, "2": {"start_sp": 8, "end_sp": 10}}}}}
    chunks = [Chunk("a", "d", "x", "", "abcde", 0, 5, 1),
              Chunk("b", "d", "x", "", "efghi", 4, 9, 1),
              Chunk("c", "d", "x", "", "ij", 8, 10, 1)]
    hits, missing = map_references("d", [{"doc_id": "x", "id_sp": "1"},
                                        {"doc_id": "x", "id_sp": "2"}], raw, chunks)
    assert hits == {"a", "b", "c"} and missing == []


def test_tied_scores_cannot_create_fake_operating_point():
    curve = risk_coverage([1, 1, 0], [False, True, False])
    assert [p["accepted"] for p in curve] == [0, 2, 3]
    assert curve[0]["risk"] is None


def test_no_safe_threshold_becomes_deny_all():
    gate = select_gate([1, 0], [True, True], method="R2", artifact_version="v", split="dev")
    assert gate.threshold is None and not gate.accepts(100)


def test_test_split_cannot_fit_threshold():
    with pytest.raises(ValueError):
        select_gate([1], [False], method="R2", artifact_version="v", split="test")
