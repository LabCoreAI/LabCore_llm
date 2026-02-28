# Troubleshooting

Use this page for fast diagnosis of common setup, data, and runtime failures.
All anchors below are referenced from guide pages.

## Setup

### Torch not installed {#torch-not-installed}

Symptoms: `ModuleNotFoundError: torch`, or scripts fail on import.

Fix:

```bash
python -m pip install -e ".[torch,dev]"
```

Use the official selector if you need a specific CUDA build:
<https://pytorch.org/get-started/locally/>

### CUDA not detected {#cuda-not-detected}

Symptoms: `torch.cuda.is_available()` returns `False`, or scripts print CPU fallback warnings.

Quick check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Checks:

- NVIDIA driver installed and up to date
- Correct PyTorch build for your CUDA runtime
- Same Python environment used for install and execution

## Data and Metadata

### Meta path mismatch {#meta-path-mismatch}

Symptoms: generation/training behaves incorrectly or cannot load tokenizer metadata.

Expected mapping:

- `txt` pipeline -> `data/processed/meta.json`
- `bin` pipeline -> `data/meta.json`

### Binary shards not found {#binary-shards-not-found}

Symptoms: `Binary shards not found` during `training.data_format = "bin"`.

Fix:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format bin --output-dir data/processed
```

### Char vocab missing {#char-vocab-missing}

Symptoms: `Char tokenizer requires vocab in meta.json`.

Fix:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
```

Then pass the matching `--meta data/processed/meta.json` to generation/export commands.

## Runtime

### Out of memory {#oom-errors}

Reduce in this order:

1. `training.batch_size`
2. `model.block_size`
3. `training.gradient_accumulation_steps`
4. Model size / preset complexity

### FlashAttention not available {#flashattention-not-available}

LabCore falls back automatically:

- FlashAttention (preferred, if available)
- PyTorch SDPA fallback
- Standard causal attention fallback

### Windows path and policy issues {#windows-path-policy}

- Activate your venv before commands.
- Run commands from repository root.
- Quote paths when directories contain spaces.
- If PowerShell policy blocks scripts, use `python ...` commands directly.
