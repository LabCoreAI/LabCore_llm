import pytest

pytest.importorskip("tiktoken")

from labcore_llm.tokenizer import CharTokenizer


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer()
    tok.fit("hello world")

    text = "hello"
    ids = tok.encode(text)
    restored = tok.decode(ids)

    assert restored == text
