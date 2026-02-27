from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    @abstractmethod
    def fit(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError
