# Getting Started

This page covers environment setup and the shortest path to a successful first run.

## Prerequisites

- Python `3.11+`
- `pip`
- Optional CUDA GPU for practical training speed

Install PyTorch first from the official selector:

<https://pytorch.org/get-started/locally/>

## Installation Profiles

```bash
pip install -e .
```

Recommended for local training and development:

```bash
pip install -e ".[torch,dev]"
```

Optional extras:

```bash
pip install -e ".[hf,demo]"
pip install -e ".[gguf]"
pip install -e ".[finetune]"
```

Install all extras:

```bash
pip install -e ".[all]"
```

## Quick Smoke Test (Char Pipeline)

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
python train.py --config configs/base.toml --tokenizer char --max-iters 200
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

Expected artifacts:

- `data/processed/meta.json`
- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

## GPU/CPU Device Rules

- `train.py` accepts `--device` and falls back to CPU if CUDA is unavailable.
- `generate.py` and `demo_gradio.py` have the same fallback behavior.
- Start with `configs/base.toml` on CPU.

## Common Setup Issues

### `ModuleNotFoundError: No module named 'torch'`

Use:

```bash
pip install -e ".[torch,dev]"
```

### Dataset download errors

`scripts/prepare_data.py` uses `datasets`; tinyshakespeare has an HTTP fallback path.

## Next Step

Continue with [Data Pipeline](data-pipeline.md).

