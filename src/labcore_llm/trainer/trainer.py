# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import json
import math
from contextlib import nullcontext
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


def determine_autocast_dtype(precision: str) -> torch.dtype | None:
    dtype_map: dict[str, torch.dtype | None] = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in dtype_map:
        raise ValueError(f"Unsupported precision value: {precision!r}")
    return dtype_map[precision]


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
    precision: str = "fp32"
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    save_best: bool = True

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")

        self.precision = self.precision.lower()
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16.")
        if not isinstance(self.early_stopping, bool):
            raise ValueError("early_stopping must be a boolean.")
        if not isinstance(self.early_stopping_patience, int) or isinstance(self.early_stopping_patience, bool):
            raise ValueError("early_stopping_patience must be an integer.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1.")
        if not isinstance(self.early_stopping_min_delta, (int, float)) or self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be a float >= 0.")
        self.early_stopping_min_delta = float(self.early_stopping_min_delta)
        if not isinstance(self.save_best, bool):
            raise ValueError("save_best must be a boolean.")


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
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but unavailable, falling back to CPU.")
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.grad_accum_steps = self.config.gradient_accumulation_steps
        self.precision = self._resolve_precision(self.config.precision)
        self.autocast_dtype = determine_autocast_dtype(self.precision)
        self.autocast_enabled = self.device.type == "cuda" and self.autocast_dtype is not None
        self.use_grad_scaler = self.autocast_enabled and self.precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        self.config.device = self.device.type
        self.config.precision = self.precision

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = Path(config.meta_path)

        self.train_loader = self._build_loader(train_data)
        self.val_loader = self._build_loader(val_data)
        self._train_iter = iter(self.train_loader)
        self._val_iter = iter(self.val_loader)

    def _resolve_precision(self, requested_precision: str) -> str:
        if self.device.type != "cuda":
            if requested_precision != "fp32":
                print(f"Warning: precision={requested_precision} requested on {self.device.type}; using fp32.")
            return "fp32"

        if requested_precision == "bf16":
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            if callable(bf16_supported) and not bf16_supported():
                print("Warning: bf16 not supported on this GPU, falling back to fp16.")
                return "fp16"

        return requested_precision

    def _autocast_context(self):
        if not self.autocast_enabled or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

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
                with self._autocast_context():
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

    def save_checkpoint(self, step: int, losses: dict[str, float] | None = None, filename: str = "ckpt_last.pt") -> Path:
        ckpt_path = self.checkpoint_dir / filename
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
        effective_batch_size = self.config.batch_size * self.grad_accum_steps
        best_val_loss = float("inf")
        patience_counter = 0
        print(
            "training setup: "
            f"batch_size={self.config.batch_size} "
            f"grad_accum_steps={self.grad_accum_steps} "
            f"effective_batch_size={effective_batch_size}"
        )
        print(f"Precision: {self.precision} | Autocast enabled: {self.autocast_enabled}")

        self.optimizer.zero_grad(set_to_none=True)
        for step in range(self.config.max_iters):
            lr = self._get_lr(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            loss_value = 0.0
            for _ in range(self.grad_accum_steps):
                xb, yb = self._next_batch("train")
                with self._autocast_context():
                    _, loss = self.model(xb, yb)
                    loss_value += loss.item()
                    scaled_loss = loss / self.grad_accum_steps
                if self.use_grad_scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if self.config.grad_clip > 0:
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if step % self.config.log_interval == 0:
                avg_loss = loss_value / self.grad_accum_steps
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

                current_val_loss = float(losses["val"])
                improved = current_val_loss < (best_val_loss - self.config.early_stopping_min_delta)
                if improved:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    if self.config.save_best:
                        best_ckpt = self.save_checkpoint(step, losses, filename="ckpt_best.pt")
                        print(f"New best model (val_loss={current_val_loss:.4f}) saved to {best_ckpt.as_posix()}")
                else:
                    patience_counter += 1

                if self.config.early_stopping and patience_counter >= self.config.early_stopping_patience:
                    print("Early stopping triggered.")
                    print(f"Early stopping at iter {step}")
                    break
            elif should_save:
                ckpt = self.save_checkpoint(step)
                print(f"checkpoint step {step}: checkpoint={ckpt.as_posix()}")
