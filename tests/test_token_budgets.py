from pathlib import Path
import pytest
from supportlens.text import JsonTokenizer, build_query, chunk_document


def test_current_question_survives_long_history(tokenizer):
    query = build_query("current question", [{"role": "user", "content": "old " * 500}], tokenizer)
    assert query.text.startswith("current question")
    assert query.truncated and not query.current_question_truncated
    assert query.used_tokens <= 256


def test_recent_history_precedes_old_history(tokenizer):
    query = build_query("current question", [{"role": "user", "content": "old"},
                                              {"role": "assistant", "content": "recent"}], tokenizer)
    assert query.text.index("recent") < query.text.index("old")


def test_overlong_current_question_is_reported(tokenizer):
    query = build_query("current " * 300, [], tokenizer)
    assert query.current_question_truncated and query.used_tokens <= 256


def test_chunk_text_seen_by_every_retriever_fits_encoder(tokenizer):
    text = "alpha beta gamma " * 250
    chunks = chunk_document("domain", "doc", "title", text, tokenizer)
    assert len(chunks) > 1
    assert all(tokenizer.count(c.text) <= 220 for c in chunks)
    assert all(text[c.start:c.end] in c.text for c in chunks)
    assert all(a.end > b.start for a, b in zip(chunks, chunks[1:]))


def test_real_minilm_tokenizer_if_available():
    path = Path("data/tokenizers/minilm/tokenizer.json")
    if not path.exists():
        pytest.skip("run scripts/fetch_tokenizers.py for the real-tokenizer integration check")
    tokenizer = JsonTokenizer(path)
    query = build_query("How do I replace my card?", [{"role": "user", "content": "old " * 1000}], tokenizer)
    assert query.text.startswith("How do I replace my card?")
    assert tokenizer.count(query.text) <= 256
    assert all(c.tokens <= 220 for c in chunk_document("d", "x", "A title", "banking words " * 500, tokenizer))
