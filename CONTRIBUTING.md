# Contributing to LabCore LLM

Thanks for your interest in contributing.

## Development Setup

1. Fork and clone the repository.
2. Create and activate a virtual environment.
3. Install dev dependencies.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Project Structure

- `src/labcore_llm/`: core package (model, tokenizer, trainer, data, config)
- `scripts/`: utilities (prepare/export/quantize/fine-tune)
- `configs/`: training presets
- `tests/`: test suite

## Local Validation Before Opening a PR

Run tests:

```bash
python -m pytest -q
```

Run lint checks used by CI:

```bash
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

## Contribution Guidelines

- Keep changes focused and atomic.
- Preserve backward compatibility unless explicitly discussed.
- Update documentation when behavior or commands change.
- Add or update tests for bug fixes and new features when relevant.
- Do not commit large model/data artifacts.

## Commit and PR Guidelines

- Use clear commit messages in imperative style.
- Open one pull request per logical change.
- Include:
  - What changed
  - Why it changed
  - How it was tested

## Reporting Bugs

Use the bug issue template and provide:

- OS and Python version
- Exact command used
- Full error trace
- Minimal reproduction steps

## Code of Conduct

By participating, you agree to follow the project Code of Conduct:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
