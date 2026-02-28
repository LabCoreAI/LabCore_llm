# Troubleshooting

Common setup and runtime issues with quick fixes.

## Installing PyTorch (CPU/CUDA)

Use the official selector for the right command on your OS and CUDA version:

<https://pytorch.org/get-started/locally/>

Typical pattern:

```bash
pip install -e ".[torch]"
```

## CUDA Not Detected (`torch.cuda.is_available() == False`)

Check:

- Your installed PyTorch build matches your CUDA runtime/driver.
- NVIDIA driver is installed and up to date.
- You are running the same Python/venv where PyTorch was installed.

Quick check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## OOM: What to Lower First

When you hit out-of-memory errors, reduce in this order:

1. `training.batch_size`
2. `model.block_size` (sequence length)
3. `training.gradient_accumulation_steps`
4. Precision strategy (if you introduce mixed precision in your environment)

If needed, move to a smaller preset model size.

## `meta.json` Location Confusion

`meta.json` location depends on output format and output directory:

- `txt` pipeline: usually `data/processed/meta.json`
- `bin` pipeline: usually `data/meta.json`

If `train.py` cannot infer metadata, verify `processed_dir` in your TOML config and the actual output path from `prepare_data.py`.

## FlashAttention Not Available

LabCore falls back automatically:

- Preferred: FlashAttention kernel (when available)
- Fallback: PyTorch SDPA
- Last path: standard causal attention

You can keep `use_flash = true`; runtime fallback handles unsupported environments.

## Windows Notes

- Use an activated venv before all commands.
- Prefer quoted paths when directories include spaces.
- Run commands from the repository root to keep relative paths stable.
- If execution policy blocks scripts, use direct `python ...` commands.
