import tomllib
from pathlib import Path

from labcore_llm.config import load_config


def test_base_config_has_required_sections():
    config_path = Path("configs/base.toml")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

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
    assert config["generation"]["top_p"] == 1.0
    assert config["generation"]["repetition_penalty"] == 1.0


def test_load_config_grad_accum_alias(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[training]\ngradient_accumulation_steps = 3\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["training"]["grad_accum_steps"] == 3
    assert config["training"]["gradient_accumulation_steps"] == 3
