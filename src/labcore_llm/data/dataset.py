from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, text: str, tokenizer: Any, block_size: int) -> None:
        token_ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
        if token_ids.ndim != 1:
            raise ValueError("CharDataset expects a 1D token stream.")
        if len(token_ids) <= block_size + 1:
            raise ValueError("Dataset too small for configured block_size.")
        self.tokens = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.tokens[idx : idx + self.block_size].copy())
        y = torch.from_numpy(self.tokens[idx + 1 : idx + 1 + self.block_size].copy())
        return x, y


class BinDataset(Dataset):
    def __init__(self, bin_path: str | Path, meta_path: str | Path, block_size: int) -> None:
        self.bin_path = Path(bin_path)
        self.meta_path = Path(meta_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Missing binary shard: {self.bin_path.as_posix()}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing meta file: {self.meta_path.as_posix()}")

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        dtype_name = meta.get("dtype", "uint16")
        np_dtype = np.dtype(dtype_name)
        self.tokens = np.memmap(self.bin_path, dtype=np_dtype, mode="r")
        if self.tokens.ndim != 1:
            self.tokens = self.tokens.reshape(-1)
        if len(self.tokens) <= block_size + 1:
            raise ValueError("Binary dataset too small for configured block_size.")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # mmap slice + copy keeps reads deterministic and avoids pinned references.
        x_np = np.asarray(self.tokens[idx : idx + self.block_size], dtype=np.int64).copy()
        y_np = np.asarray(self.tokens[idx + 1 : idx + 1 + self.block_size], dtype=np.int64).copy()
        return torch.from_numpy(x_np), torch.from_numpy(y_np)
