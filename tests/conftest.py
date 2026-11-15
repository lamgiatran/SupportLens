from pathlib import Path
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

from supportlens.text import JsonTokenizer


@pytest.fixture
def tokenizer(tmp_path):
    # Offline WordPiece fixture; production tokenizer is tested separately if present.
    vocab = {word: i for i, word in enumerate([
        "[UNK]", "[CLS]", "[SEP]", "current", "question", "old", "recent", "answer",
        "user", "assistant", ":", "first", "second", "third", "title", "alpha", "beta", "gamma"
    ])}
    t = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    t.normalizer = BertNormalizer(lowercase=True)
    t.pre_tokenizer = BertPreTokenizer()
    t.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    file = tmp_path / "tokenizer.json"
    t.save(str(file))
    return JsonTokenizer(file)


@pytest.fixture
def rows():
    return [
        {"domain": "synthetic", "doc_id": "a", "title": "Address", "text": "uniqueaddress change procedure"},
        {"domain": "synthetic", "doc_id": "b", "title": "Cars", "text": "vehicle registration"},
        {"domain": "synthetic", "doc_id": "c", "title": "Forms", "text": "student enrollment"},
    ]
