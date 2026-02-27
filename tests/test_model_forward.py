import torch

from labcore_llm.model import GPT, GPTConfig


def test_model_forward_shapes_and_loss():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(x, y)

    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
