.PHONY: install prepare train generate test clean
PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .[dev]

prepare:
	$(PYTHON) -m scripts.prepare_data --dataset tinyshakespeare --tokenizer char --output-format txt

train:
	$(PYTHON) train.py --config configs/base.toml

generate:
	$(PYTHON) generate.py --prompt "To be, or not to be"

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['runs', 'checkpoints', 'outputs', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist', '__pycache__']]"
