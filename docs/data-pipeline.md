# Data Pipeline

Use this page to prepare training data and metadata in a predictable layout.
Prerequisite: dependencies installed from [Getting Started](getting-started.md).

## Command(s)

Reference `txt` pipeline (used by `configs/base.toml`):

```bash
python scripts/prepare_data.py \
  --dataset tinyshakespeare \
  --tokenizer char \
  --output-format txt \
  --raw-dir data/raw \
  --output-dir data/processed \
  --val-ratio 0.1
```

Alternative `bin` pipeline:

```bash
python scripts/prepare_data.py \
  --dataset tinyshakespeare \
  --tokenizer char \
  --output-format bin \
  --raw-dir data/raw \
  --output-dir data/processed \
  --val-ratio 0.1
```

## Output Files / Artifacts Produced

`txt` format (`output-dir = data/processed`):

- `data/processed/train.txt`
- `data/processed/val.txt`
- `data/processed/corpus.txt`
- `data/processed/train.npy`
- `data/processed/val.npy`
- `data/processed/meta.json` (`META_TXT`)

`bin` format:

- `data/train.bin`
- `data/val.bin`
- `data/meta.json` (`META_BIN`)

!!! note
    For `--output-format bin`, if `--output-dir` ends with `processed`, binary files are written to its parent (`data/`).

## Format Selection

- Use `txt` when training with `training.data_format = "txt"` and metadata at `data/processed/meta.json`.
- Use `bin` when training with `training.data_format = "bin"` and metadata at `data/meta.json`.

## Common Errors

- Missing binary shards: see [Binary shards not found](troubleshooting.md#binary-shards-not-found).
- Wrong metadata path: see [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Char tokenizer vocab issues: see [Char vocab missing](troubleshooting.md#char-vocab-missing).

## Next / Related

- [Training](training.md)
- [Inference & Demo](inference-and-demo.md)
- [Troubleshooting](troubleshooting.md)
