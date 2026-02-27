import tomllib
from pathlib import Path


def test_base_config_has_required_sections():
    config_path = Path("configs/base.toml")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    assert "data" in config
    assert "model" in config
    assert "training" in config
    assert config["model"]["block_size"] > 0
