import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/eval_claim_citations.py"
spec = importlib.util.spec_from_file_location("citation_experiment", path)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_quote_must_match_its_own_visible_source():
    sources = {"[S1]": "A form has four sections.", "[S2]": "Other text."}
    line = "[S2] | A form has four sections. | The form has four sections."
    assert experiment.validate_quotes(line, sources)[0] == "quote_mismatch"
    assert experiment.validate_quotes(line.replace("S2", "S1"), sources)[0] == "answer"
    assert experiment.validate_quotes("PASS", sources) == ("refused", [])


def test_selector_returns_exact_visible_offsets():
    sources = {"[S2]": "A heading\n  The following fees are waived. Unfinished tail"}
    status, [claim] = experiment.resolve_sentence('{"id":1}', sources)
    assert status == "answer" and claim["claim"] == "The following fees are waived."
    assert sources[claim["source"]][claim["start"]:claim["end"]] == claim["claim"]


@pytest.mark.parametrize("raw,status", [('{"id":0}', "refused"), ('{"id":true}', "format_error"),
                                        ('{"id":-1}', "unknown_id"), ('{"id":7}', "unknown_id"),
                                        ('{"id":2,"claim":"invented"}', "format_error")])
def test_selector_cannot_publish_unknown_ids_or_injected_text(raw, status):
    assert experiment.resolve_sentence(raw, {}) == (status, [])


def test_paired_samples_exclude_previously_seen_dialogues():
    queries = [{"query_id": f"{d}:{i}", "dialogue_id": str(d), "valid_input": True}
               for d in range(8) for i in range(2)]
    first = experiment.one_per_dialogue(queries, {"0"}, 3)
    excluded = {"0"} | {r["dialogue_id"] for r in first}
    second = experiment.one_per_dialogue(queries[::-1], excluded, 4)
    assert len({r["dialogue_id"] for r in second} - excluded) == 4
