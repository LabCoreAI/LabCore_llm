from __future__ import annotations

import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        cfg = tomllib.load(handle)

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

    return cfg
