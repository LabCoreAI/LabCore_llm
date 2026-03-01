from __future__ import annotations

import torch
import torch.nn.functional as F

from labcore_llm.model import GPT, GPTConfig
from labcore_llm.model.gpt import apply_repetition_penalty


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
