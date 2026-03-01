from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from generate import configure_generation_reproducibility
from labcore_llm.model import GPT, GPTConfig
from labcore_llm.model.gpt import apply_repetition_penalty


def _is_warn_only_enabled() -> bool:
    fn = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return bool(fn()) if callable(fn) else False


def _build_model() -> GPT:
    cfg = GPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    model = GPT(cfg)
    model.eval()
    return model


def test_generate_top_p_one_matches_baseline():
    model = _build_model()
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    torch.manual_seed(123)
    y_baseline = model.generate(x.clone(), max_new_tokens=8, temperature=0.9, top_k=12)
    torch.manual_seed(123)
    y_top_p = model.generate(x.clone(), max_new_tokens=8, temperature=0.9, top_k=12, top_p=1.0)

    assert torch.equal(y_baseline, y_top_p)


def test_generate_top_p_less_than_one_runs():
    model = _build_model()
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    torch.manual_seed(123)
    y = model.generate(x, max_new_tokens=8, temperature=0.9, top_k=12, top_p=0.9)

    assert y.shape == (1, 12)


def test_generate_repetition_penalty_one_matches_baseline():
    model = _build_model()
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    torch.manual_seed(456)
    y_baseline = model.generate(x.clone(), max_new_tokens=8, temperature=0.9, top_k=12)
    torch.manual_seed(456)
    y_penalty = model.generate(x.clone(), max_new_tokens=8, temperature=0.9, top_k=12, repetition_penalty=1.0)

    assert torch.equal(y_baseline, y_penalty)


def test_repetition_penalty_reduces_repeated_token_probability():
    logits = torch.tensor([[4.0, 1.0, 0.5]], dtype=torch.float32)
    generated_tokens = torch.tensor([[0, 0, 2]], dtype=torch.long)

    probs_before = F.softmax(logits, dim=-1)
    penalized_logits = apply_repetition_penalty(logits.clone(), generated_tokens, repetition_penalty=1.5)
    probs_after = F.softmax(penalized_logits, dim=-1)

    assert probs_after[0, 0] < probs_before[0, 0]


def test_generation_seed_reproduces_same_output():
    model = _build_model()
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    configure_generation_reproducibility(seed=1337, deterministic=False)
    y1 = model.generate(x.clone(), max_new_tokens=8, temperature=0.9, top_k=12, top_p=0.9, repetition_penalty=1.1)
    configure_generation_reproducibility(seed=1337, deterministic=False)
    y2 = model.generate(x.clone(), max_new_tokens=8, temperature=0.9, top_k=12, top_p=0.9, repetition_penalty=1.1)

    assert torch.equal(y1, y2)


def test_generation_without_seed_does_not_override_rng_state():
    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)
    baseline = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.seed(21)
    np.random.seed(21)
    torch.manual_seed(21)
    configure_generation_reproducibility(seed=None, deterministic=False)
    after = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert after == baseline


def test_generation_deterministic_mode_cpu_runs():
    prev_enabled = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = _is_warn_only_enabled()
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        configure_generation_reproducibility(seed=7, deterministic=True)
        model = _build_model()
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        y = model.generate(x, max_new_tokens=4, temperature=0.9, top_k=8)

        assert y.shape == (1, 8)
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(prev_enabled, warn_only=prev_warn_only)
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark


def test_generation_deterministic_mode_cuda_runs_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    prev_enabled = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = _is_warn_only_enabled()
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        configure_generation_reproducibility(seed=7, deterministic=True)
        model = _build_model().to("cuda")
        x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")
        y = model.generate(x, max_new_tokens=4, temperature=0.9, top_k=8)

        assert y.shape == (1, 8)
    finally:
        torch.use_deterministic_algorithms(prev_enabled, warn_only=prev_warn_only)
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark
