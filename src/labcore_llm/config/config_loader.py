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

    return cfg
