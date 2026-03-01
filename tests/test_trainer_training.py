# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import numpy as np
import pytest
import torch

from labcore_llm.model import GPT, GPTConfig
from labcore_llm.trainer import Trainer, TrainerConfig


def _build_trainer(
    tmp_path,
    *,
    grad_accum_steps: int = 1,
    precision: str = "fp32",
    device: str = "cpu",
    max_iters: int = 2,
) -> Trainer:
    model_cfg = GPTConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    runtime_device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = GPT(model_cfg).to(runtime_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_tokens = np.arange(0, 256, dtype=np.int64) % model_cfg.vocab_size
    val_tokens = np.arange(128, 384, dtype=np.int64) % model_cfg.vocab_size

    trainer_cfg = TrainerConfig(
        batch_size=4,
        block_size=model_cfg.block_size,
        max_iters=max_iters,
        eval_interval=max_iters,
        eval_iters=1,
        log_interval=max_iters + 1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        gradient_accumulation_steps=grad_accum_steps,
        save_interval=max_iters,
        device=device,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        precision=precision,
    )
    return Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_tokens,
        val_data=val_tokens,
        config=trainer_cfg,
    )


@pytest.mark.parametrize("grad_accum_steps", [1, 2])
def test_grad_accum_steps_training_runs(tmp_path, grad_accum_steps):
    trainer = _build_trainer(tmp_path, grad_accum_steps=grad_accum_steps, max_iters=2)
    trainer.train()

    assert (tmp_path / "checkpoints" / "ckpt_last.pt").exists()


def test_optimizer_step_count_matches_training_iterations(tmp_path):
    trainer = _build_trainer(tmp_path, grad_accum_steps=2, max_iters=4)
    step_calls = 0
    original_step = trainer.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counted_step  # type: ignore[method-assign]
    trainer.train()

    assert step_calls == trainer.config.max_iters


def test_precision_fp32_cpu_runs(tmp_path):
    trainer = _build_trainer(tmp_path, precision="fp32", device="cpu", max_iters=1)
    trainer.train()

    assert trainer.precision == "fp32"
    assert trainer.autocast_enabled is False


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_precision_cuda_modes(precision, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    trainer = _build_trainer(tmp_path, precision=precision, device="cuda", max_iters=1)
    trainer.train()

    assert trainer.autocast_enabled is True
    if precision == "fp16":
        assert trainer.use_grad_scaler is True
    else:
        assert trainer.precision in {"bf16", "fp16"}
