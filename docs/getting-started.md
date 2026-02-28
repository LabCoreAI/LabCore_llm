# Getting Started

Use this page to get a reproducible first run in about 5 minutes and confirm your environment is healthy.
Prerequisites: Python `3.11+`, `pip`, and optional CUDA GPU.

## Command(s)

Install dependencies:

```bash
python -m pip install -e ".[torch,dev]"
```

Check CUDA visibility:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Quick Start (reference preset: tinyshakespeare + char + `configs/base.toml`):

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

!!! tip
    For a strict 5-minute smoke test, start the training command, wait for the first eval/checkpoint output, then stop and run generation.

## Output Files / Artifacts Produced

- `data/processed/meta.json`
- `data/processed/train.txt`, `data/processed/val.txt`
- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

## Success Checklist

- Training logs show at least two `train_loss` lines and the value trends down.
- A checkpoint exists at `checkpoints/ckpt_last.pt`.
- `generate.py` returns non-empty text from your prompt.

## Common Errors

- `ModuleNotFoundError: torch`: see [Torch not installed](troubleshooting.md#torch-not-installed).
- CUDA expected but disabled: see [CUDA not detected](troubleshooting.md#cuda-not-detected).
- Metadata mismatch: see [Meta path mismatch](troubleshooting.md#meta-path-mismatch).

!!! warning
    Keep `--checkpoint` and `--meta` aligned with the same run. Mixed files from different runs produce misleading results.

## Next / Related

- [Data Pipeline](data-pipeline.md)
- [Training](training.md)
- [Troubleshooting](troubleshooting.md)
