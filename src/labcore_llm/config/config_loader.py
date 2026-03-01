# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import re
import tomllib
from pathlib import Path


NULL_SENTINEL = "__LABCORE_NULL__"
SEED_NULL_PATTERN = re.compile(r'(?m)^(\s*seed\s*=\s*)null(\s*(?:#.*)?)$')


def _replace_null_sentinel(value):
    if isinstance(value, dict):
        return {k: _replace_null_sentinel(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_null_sentinel(v) for v in value]
    if value == NULL_SENTINEL:
        return None
    return value


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    normalized = SEED_NULL_PATTERN.sub(rf'\1"{NULL_SENTINEL}"\2', raw)
    cfg = _replace_null_sentinel(tomllib.loads(normalized))

    general_cfg = cfg.setdefault("general", {})
    general_cfg.setdefault("tokenizer", "char")

    data_cfg = cfg.setdefault("data", {})
    data_cfg.setdefault("processed_dir", "data/processed")

    model_cfg = cfg.setdefault("model", {})
    model_cfg.setdefault("use_rope", False)
    model_cfg.setdefault("use_flash", False)

    training_cfg = cfg.setdefault("training", {})
    training_cfg.setdefault("data_format", "txt")
    training_cfg.setdefault("grad_accum_steps", training_cfg.get("gradient_accumulation_steps", 1))
    training_cfg.setdefault("gradient_accumulation_steps", training_cfg["grad_accum_steps"])
    training_cfg.setdefault("precision", "fp32")

    generation_cfg = cfg.setdefault("generation", {})
    generation_cfg.setdefault("top_p", 1.0)
    generation_cfg.setdefault("repetition_penalty", 1.0)
    generation_cfg.setdefault("seed", None)
    generation_cfg.setdefault("deterministic", False)

    seed = generation_cfg["seed"]
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("generation.seed must be an integer or null.")
    if not isinstance(generation_cfg["deterministic"], bool):
        raise ValueError("generation.deterministic must be a boolean.")

    return cfg
