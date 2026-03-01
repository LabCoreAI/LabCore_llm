# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from pathlib import Path

from labcore_llm.config import load_config


def test_base_config_has_required_sections():
    config = load_config(Path("configs/base.toml"))

    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert config["model"]["block_size"] > 0


def test_load_config_training_defaults(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[training]\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["training"]["grad_accum_steps"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["precision"] == "fp32"
    assert config["training"]["early_stopping"] is False
    assert config["training"]["early_stopping_patience"] == 5
    assert config["training"]["early_stopping_min_delta"] == 0.0
    assert config["training"]["save_best"] is True
    assert config["generation"]["top_p"] == 1.0
    assert config["generation"]["repetition_penalty"] == 1.0
    assert config["generation"]["use_kv_cache"] is True
    assert config["generation"]["stream"] is True
    assert config["generation"]["system_prompt"] == ""
    assert config["generation"]["max_history_turns"] == 6
    assert config["generation"]["seed"] is None
    assert config["generation"]["deterministic"] is False


def test_load_config_grad_accum_alias(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[training]\ngradient_accumulation_steps = 3\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["training"]["grad_accum_steps"] == 3
    assert config["training"]["gradient_accumulation_steps"] == 3


def test_load_config_generation_seed_null_literal(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[generation]\nseed = null\ndeterministic = true\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["generation"]["seed"] is None
    assert config["generation"]["deterministic"] is True
