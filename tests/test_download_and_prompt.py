import io
import json
import zipfile
import pytest
from supportlens.download import extract_multidoc, sha256, write_manifest, verify_manifest
from supportlens.generation import make_prompt


def test_archive_paths_never_escape_output(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../multidoc2dial_doc.json", json.dumps({"doc_data": {}}))
        z.writestr("ok/multidoc2dial_dial_train.json", json.dumps({"dial_data": {}}))
        z.writestr("ok/multidoc2dial_dial_validation.json", json.dumps({"dial_data": {}}))
        z.writestr("../../unexpected.txt", "do not extract")
    out = tmp_path / "safe"
    files = extract_multidoc(archive, out)
    assert len(files) == 3 and all(p.parent == out for p in files)
    assert not (tmp_path / "unexpected.txt").exists()


def test_manifest_detects_changed_bytes(tmp_path):
    file = tmp_path / "data.txt"
    file.write_text("original")
    manifest = write_manifest(tmp_path, [{"path": "data.txt", "url": "https://example.invalid/data"}], tmp_path / "manifest.json")
    file.write_text("changed")
    with pytest.raises(ValueError):
        verify_manifest(tmp_path, manifest)


def test_prompt_packs_evidence_not_system_or_question(tokenizer):
    prompt = make_prompt("current question", [{"evidence_id": "a", "text": "alpha " * 1000}], tokenizer,
                         num_ctx=256, num_predict=20, evidence_budget=40)
    assert prompt.evidence_tokens <= 40
    assert "current question" in prompt.text and "Answer only" in prompt.text
    assert prompt.tokens + 20 + 32 <= 256


def test_context_overflow_is_an_error_not_silent_truncation(tokenizer):
    with pytest.raises(ValueError):
        make_prompt("current " * 300, [], tokenizer, num_ctx=256, num_predict=20)
