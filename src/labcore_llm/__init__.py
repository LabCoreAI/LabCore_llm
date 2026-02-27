from .config import load_config
from .model import GPT, GPTConfig
from .trainer import Trainer, TrainerConfig

__all__ = [
    "load_config",
    "GPT",
    "GPTConfig",
    "Trainer",
    "TrainerConfig",
    "CharTokenizer",
    "BPETokenizer",
    "BaseTokenizer",
]


def __getattr__(name: str):
    if name == "CharTokenizer":
        from .tokenizer.char_tokenizer import CharTokenizer

        return CharTokenizer
    if name == "BPETokenizer":
        from .tokenizer.bpe_tokenizer import BPETokenizer

        return BPETokenizer
    if name == "BaseTokenizer":
        from .tokenizer.base import BaseTokenizer

        return BaseTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
