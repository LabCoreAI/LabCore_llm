import tiktoken
from typing import List


class BPETokenizer:
    """Tokenizer BPE compatible avec l'interface de CharTokenizer."""

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
        self.encoding_name = encoding_name

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.encoding.decode(ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, batch: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in batch]

    @property
    def eos_token_id(self) -> int:
        return self.encoding.eot_token

    @property
    def bos_token_id(self) -> int:
        return self.encoding.eot_token
