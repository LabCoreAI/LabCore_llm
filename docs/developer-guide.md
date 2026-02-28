# Developer Guide

This page is for contributors working on the codebase itself.

## Repository Map

```text
src/labcore_llm/
  config/      # TOML loader and defaults
  data/        # dataset abstractions
  model/       # GPT model implementation
  tokenizer/   # char + BPE tokenizers
  trainer/     # training loop, scheduler, checkpointing

scripts/       # data prep, export, quantize, fine-tune helpers
configs/       # TOML presets
tests/         # unit tests
```

## Local Dev Environment

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[torch,dev]"
```

## Validation Commands

Run tests:

```bash
python -m pytest -q
```

Run lint rules aligned with CI:

```bash
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

## CI Workflows

- `.github/workflows/ci.yml`: lint + tests
- `.github/workflows/docs.yml`: MkDocs build and deploy

## Contribution Quality Bar

- Keep commits focused and atomic.
- Update docs for behavior/CLI changes.
- Add tests for bug fixes and new logic.
- Do not commit large data/model artifacts.

## Packaging Notes

- Project uses `src/` layout with setuptools.
- Optional dependency groups are defined in `pyproject.toml`.
- Entry scripts are plain Python files, not console-script wrappers.

