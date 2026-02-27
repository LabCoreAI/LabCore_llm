from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.dataset import BinDataset, CharDataset
from ..model import GPT


class ArrayDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, block_size: int) -> None:
        tokens = np.asarray(token_ids, dtype=np.int64)
        if tokens.ndim != 1:
            raise ValueError("ArrayDataset expects a 1D token stream.")
        if len(tokens) <= block_size + 1:
            raise ValueError("Dataset too small for configured block_size.")
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.tokens[idx : idx + self.block_size].copy())
        y = torch.from_numpy(self.tokens[idx + 1 : idx + 1 + self.block_size].copy())
        return x, y


@dataclass
class TrainerConfig:
    batch_size: int = 32
    block_size: int = 128
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_iters: int = 0
    lr_decay_iters: int = 2000
    min_lr: float = 3e-5
    save_interval: int = 500
    beta1: float = 0.9
    beta2: float = 0.95
    tokenizer_name: str = "char"
    data_format: str = "txt"
    meta_path: str = "data/processed/meta.json"
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"


class Trainer:
    def __init__(
        self,
        model: GPT,
        optimizer: torch.optim.Optimizer,
        train_data: np.ndarray | str | Path,
        val_data: np.ndarray | str | Path,
        config: TrainerConfig,
        tokenizer: Any | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = Path(config.meta_path)

        self.train_loader = self._build_loader(train_data)
        self.val_loader = self._build_loader(val_data)
        self._train_iter = iter(self.train_loader)
        self._val_iter = iter(self.val_loader)

    def _build_loader(self, data: np.ndarray | str | Path) -> DataLoader:
        if self.config.data_format == "bin":
            dataset: Dataset = BinDataset(bin_path=Path(data), meta_path=self.meta_path, block_size=self.config.block_size)
        elif isinstance(data, np.ndarray):
            dataset = ArrayDataset(data, block_size=self.config.block_size)
        else:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required when training from text datasets.")
            dataset = CharDataset(str(data), tokenizer=self.tokenizer, block_size=self.config.block_size)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _next_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            iterator = self._train_iter
            loader = self.train_loader
        else:
            iterator = self._val_iter
            loader = self.val_loader

        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        if split == "train":
            self._train_iter = iterator
        else:
            self._val_iter = iterator

        return x.to(self.device), y.to(self.device)

    def _get_lr(self, step: int) -> float:
        if step < self.config.warmup_iters and self.config.warmup_iters > 0:
            return self.config.learning_rate * (step + 1) / self.config.warmup_iters
        if step >= self.config.lr_decay_iters:
            return self.config.min_lr
        if self.config.lr_decay_iters <= self.config.warmup_iters:
            return self.config.min_lr

        decay_ratio = (step - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                xb, yb = self._next_batch(split)
                _, loss = self.model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def _tokenizer_meta(self) -> dict[str, Any]:
        if self.tokenizer is None:
            return {"type": self.config.tokenizer_name}
        if self.config.tokenizer_name == "bpe":
            return {
                "type": "bpe",
                "encoding_name": getattr(self.tokenizer, "encoding_name", "gpt2"),
            }
        return {
            "type": "char",
            "vocab": getattr(self.tokenizer, "vocab", []),
        }

    def save_checkpoint(self, step: int, losses: dict[str, float] | None = None) -> Path:
        ckpt_path = self.checkpoint_dir / "ckpt_last.pt"
        if losses is None:
            losses = {}
        payload = {
            "step": step,
            "model_config": self.model.config_dict,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_config": asdict(self.config),
            "losses": losses,
            "tokenizer": self._tokenizer_meta(),
        }
        torch.save(payload, ckpt_path)

        meta_path = self.checkpoint_dir / "train_log.json"
        meta = {
            "step": step,
            "losses": losses,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return ckpt_path

    def train(self) -> None:
        self.model.train()
        for step in range(self.config.max_iters):
            lr = self._get_lr(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(self.config.gradient_accumulation_steps):
                xb, yb = self._next_batch("train")
                _, loss = self.model(xb, yb)
                (loss / self.config.gradient_accumulation_steps).backward()
                loss_value += loss.item()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            if step % self.config.log_interval == 0:
                avg_loss = loss_value / self.config.gradient_accumulation_steps
                print(f"step {step}: train_loss={avg_loss:.4f} lr={lr:.6f}")

            should_eval = step % self.config.eval_interval == 0 or step == self.config.max_iters - 1
            should_save = step % self.config.save_interval == 0 or step == self.config.max_iters - 1

            if should_eval:
                losses = self.estimate_loss()
                ckpt = self.save_checkpoint(step, losses)
                print(
                    f"eval step {step}: train={losses['train']:.4f} val={losses['val']:.4f} "
                    f"checkpoint={ckpt.as_posix()}"
                )
            elif should_save:
                ckpt = self.save_checkpoint(step)
                print(f"checkpoint step {step}: checkpoint={ckpt.as_posix()}")
