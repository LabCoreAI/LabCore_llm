from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseTokenizer


@dataclass
class CharTokenizer(BaseTokenizer):
    vocab: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, text: str) -> None:
        self.vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        missing = [ch for ch in text if ch not in self.stoi]
        if missing:
            sample = "".join(sorted(set(missing))[:5])
            raise ValueError(f"Unknown characters found in input: {sample!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[i] for i in token_ids)

    def to_dict(self) -> dict:
        return {"type": "char", "vocab": self.vocab}

    @classmethod
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        return cls(vocab=payload["vocab"])
