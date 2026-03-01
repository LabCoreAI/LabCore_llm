# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


KVCache = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[KVCache, ...]


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
        self.block_size = config.block_size
        self.use_rope = config.use_rope
        self.use_flash = config.use_flash

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = config.dropout

        head_dim = config.n_embd // config.n_head
        self.rope = RotaryEmbedding(head_dim=head_dim) if self.use_rope else None

    def _flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
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
                    is_causal=is_causal,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_len: int,
        key_len: int,
        past_len: int,
    ) -> torch.Tensor:
        head_dim = q.size(-1)
        att = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)

        query_positions = torch.arange(past_len, past_len + query_len, device=q.device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=q.device).unsqueeze(0)
        causal_mask = key_positions <= query_positions
        att = att.masked_fill(~causal_mask.view(1, 1, query_len, key_len), float("-inf"))

        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.attn_dropout, training=self.training)
        return att @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_kv(x, past_kv=None, use_cache=False)
        return y

    def forward_with_kv(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        bsz, seqlen, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_dim = channels // self.n_head
        q = q.view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)
        k_new = k.view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)
        v_new = v.view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            k_cache, v_cache = past_kv
            if k_cache.size(0) != bsz or v_cache.size(0) != bsz:
                raise ValueError("Batch size mismatch between input and KV cache.")
            past_len = k_cache.size(2)
        else:
            k_cache = None
            v_cache = None

        if self.rope is not None:
            rope_len = past_len + seqlen
            cos, sin = self.rope.get_cos_sin(seq_len=rope_len, device=x.device, dtype=q.dtype)
            cos = cos[:, :, past_len : past_len + seqlen, :]
            sin = sin[:, :, past_len : past_len + seqlen, :]
            q = self.rope.apply_rotary(q, cos, sin)
            k_new = self.rope.apply_rotary(k_new, cos, sin)

        if k_cache is not None and v_cache is not None:
            k = torch.cat((k_cache, k_new), dim=2)
            v = torch.cat((v_cache, v_new), dim=2)
        else:
            k = k_new
            v = v_new

        if k.size(2) > self.block_size:
            k = k[:, :, -self.block_size :, :]
            v = v[:, :, -self.block_size :, :]

        key_len = k.size(2)
        effective_past_len = max(key_len - seqlen, 0)

        if hasattr(F, "scaled_dot_product_attention"):
            y = self._flash_attention(q, k, v, is_causal=(effective_past_len == 0))
        else:
            y = self._manual_attention(
                q,
                k,
                v,
                query_len=seqlen,
                key_len=key_len,
                past_len=effective_past_len,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, channels)
        y = self.resid_dropout(self.c_proj(y))
        present = (k, v) if use_cache else None
        return y, present


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

    def forward_with_kv(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present_kv = self.attn.forward_with_kv(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


def apply_top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p == 1.0:
        return logits
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be in the interval (0, 1].")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove_mask = cumulative_probs > top_p
    sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
    sorted_remove_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_remove_mask, float("-inf"))

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0 or generated_tokens.numel() == 0:
        return logits
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be > 0.")

    for batch_idx in range(logits.size(0)):
        repeated_tokens = torch.unique(generated_tokens[batch_idx])
        if repeated_tokens.numel() == 0:
            continue
        logits[batch_idx, repeated_tokens] = logits[batch_idx, repeated_tokens] / repetition_penalty
    return logits


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

    def forward_with_kv(
        self,
        idx: torch.Tensor,
        past_kv: PastKeyValues | None = None,
    ) -> tuple[torch.Tensor, PastKeyValues]:
        if past_kv is not None and len(past_kv) != len(self.h):
            raise ValueError(f"Expected {len(self.h)} KV cache entries, got {len(past_kv)}.")

        if idx.size(1) == 0:
            raise ValueError("Input sequence must contain at least one token.")
        if past_kv is not None:
            idx = idx[:, -1:]

        _, seqlen = idx.size()
        if seqlen > self.config.block_size:
            raise ValueError(f"Sequence length {seqlen} exceeds block size {self.config.block_size}")

        x = self.wte(idx)
        if self.wpe is not None:
            if past_kv is None:
                pos = torch.arange(0, seqlen, dtype=torch.long, device=idx.device)
            else:
                cached_len = past_kv[0][0].size(2) if len(past_kv) > 0 else 0
                if seqlen == 1:
                    pos = torch.tensor([min(cached_len, self.config.block_size - 1)], dtype=torch.long, device=idx.device)
                else:
                    start = max(0, min(cached_len, self.config.block_size - seqlen))
                    pos = torch.arange(start, start + seqlen, dtype=torch.long, device=idx.device)
            x = x + self.wpe(pos)

        x = self.drop(x)
        new_past: list[KVCache] = []
        for layer_idx, block in enumerate(self.h):
            layer_past = past_kv[layer_idx] if past_kv is not None else None
            x, layer_present = block.forward_with_kv(x, past_kv=layer_past, use_cache=True)
            if layer_present is None:
                raise RuntimeError("KV caching expected but missing at attention block.")
            new_past.append(layer_present)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, tuple(new_past)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        stream: bool = False,
    ) -> torch.Tensor | Iterator[int]:
        if idx.size(1) == 0:
            raise ValueError("Prompt must contain at least one token.")
        if stream and idx.size(0) != 1:
            raise ValueError("Streaming mode currently supports batch size 1 only.")

        prompt_len = idx.size(1)

        def sample_next_token(step_logits: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
            logits = step_logits[:, -1, :] / max(temperature, 1e-6)

            generated_tokens = current_tokens[:, prompt_len:]
            logits = apply_repetition_penalty(
                logits,
                generated_tokens=generated_tokens,
                repetition_penalty=repetition_penalty,
            )

            if top_k is not None and top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")
            logits = apply_top_p_filter(logits, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        def run_non_stream() -> torch.Tensor:
            idx_out = idx
            if max_new_tokens <= 0:
                return idx_out

            if not use_kv_cache:
                for _ in range(max_new_tokens):
                    idx_cond = idx_out[:, -self.config.block_size :]
                    logits, _ = self(idx_cond)
                    idx_next = sample_next_token(logits, idx_out)
                    idx_out = torch.cat((idx_out, idx_next), dim=1)
                return idx_out

            idx_cond = idx_out[:, -self.config.block_size :]
            logits, past_kv = self.forward_with_kv(idx_cond, past_kv=None)
            for step in range(max_new_tokens):
                idx_next = sample_next_token(logits, idx_out)
                idx_out = torch.cat((idx_out, idx_next), dim=1)
                if step != max_new_tokens - 1:
                    logits, past_kv = self.forward_with_kv(idx_next, past_kv=past_kv)
            return idx_out

        def run_stream() -> Iterator[int]:
            idx_out = idx
            if max_new_tokens <= 0:
                return

            if not use_kv_cache:
                for _ in range(max_new_tokens):
                    idx_cond = idx_out[:, -self.config.block_size :]
                    logits, _ = self(idx_cond)
                    idx_next = sample_next_token(logits, idx_out)
                    idx_out = torch.cat((idx_out, idx_next), dim=1)
                    yield int(idx_next[0, 0].item())
                return

            idx_cond = idx_out[:, -self.config.block_size :]
            logits, past_kv = self.forward_with_kv(idx_cond, past_kv=None)
            for step in range(max_new_tokens):
                idx_next = sample_next_token(logits, idx_out)
                idx_out = torch.cat((idx_out, idx_next), dim=1)
                yield int(idx_next[0, 0].item())
                if step != max_new_tokens - 1:
                    logits, past_kv = self.forward_with_kv(idx_next, past_kv=past_kv)

        if stream:
            return run_stream()
        return run_non_stream()
