from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    use_rope: bool = False
    use_flash: bool = False


class RotaryEmbedding(nn.Module):
    # RoPE 2026 / FlashAttention: lightweight rotary cache for decoder-only attention.
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_seq_len = 0
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needs_refresh = (
            self._cache is None
            or self._cache_seq_len < seq_len
            or self._cache[0].device != device
            or self._cache[0].dtype != dtype
        )
        if needs_refresh:
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(positions, self.inv_freq)
            cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
            sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
            self._cache_seq_len = seq_len
            self._cache = (cos, sin)
        return self._cache  # type: ignore[return-value]

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_rope = config.use_rope
        self.use_flash = config.use_flash

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = config.dropout

        head_dim = config.n_embd // config.n_head
        self.rope = RotaryEmbedding(head_dim=head_dim) if self.use_rope else None
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        # RoPE 2026 / FlashAttention
        if self.use_flash and q.is_cuda and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlen: int) -> torch.Tensor:
        head_dim = q.size(-1)
        att = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :seqlen, :seqlen] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.attn_dropout, training=self.training)
        return att @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_dim = channels // self.n_head
        q = q.view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope.get_cos_sin(seq_len=seqlen, device=x.device, dtype=q.dtype)
            q = self.rope.apply_rotary(q, cos, sin)
            k = self.rope.apply_rotary(k, cos, sin)

        if hasattr(F, "scaled_dot_product_attention"):
            y = self._flash_attention(q, k, v)
        else:
            y = self._manual_attention(q, k, v, seqlen)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd) if not config.use_rope else None
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    @property
    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seqlen = idx.size()
        if seqlen > self.config.block_size:
            raise ValueError(f"Sequence length {seqlen} exceeds block size {self.config.block_size}")

        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(0, seqlen, dtype=torch.long, device=idx.device)
            x = x + self.wpe(pos)

        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
