# Training

Use this page to launch, monitor, and checkpoint training runs with consistent settings.
Prerequisites: prepared dataset and metadata from [Data Pipeline](data-pipeline.md).

## Command(s)

Reference training command:

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
```

Device override examples:

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cpu
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cuda
```

`train.py` flags:

- `--config`: TOML preset path (`CONFIG_EXAMPLE = configs/base.toml`)
- `--max-iters`: runtime override for total iterations
- `--device`: `cpu` or `cuda`
- `--tokenizer`: `char` or `bpe`

## Precision and Gradient Accumulation

Configure these in `[training]`:

- `grad_accum_steps`: gradient accumulation factor (default `1`)
- `precision`: `fp32` (default), `fp16`, or `bf16`

`effective_batch_size = batch_size * grad_accum_steps`

Mixed precision is enabled only when `device = "cuda"` and `precision != "fp32"`.
On CPU, training falls back to `fp32`.

RTX 4060 example:

```toml
[training]
batch_size = 8
grad_accum_steps = 4
precision = "fp16"
```

### Best Checkpoint & Early Stopping

- `save_best` saves `checkpoints/ckpt_best.pt` whenever validation loss improves by at least `early_stopping_min_delta`.
- `early_stopping` is disabled by default.
- `early_stopping_patience` counts evaluation rounds without sufficient validation improvement.
- `early_stopping_min_delta` defines the minimum improvement threshold on `val_loss`.

```toml
[training]
early_stopping = true
early_stopping_patience = 3
early_stopping_min_delta = 0.001
save_best = true
```

## `data_format` and Metadata Mapping

| Training mode | Config value | Data artifacts expected | Metadata path |
|---|---|---|---|
| Text pipeline | `training.data_format = "txt"` | `data/processed/train.txt` + `data/processed/val.txt` (or `.npy`) | `data/processed/meta.json` (`META_TXT`) |
| Binary pipeline | `training.data_format = "bin"` | `data/train.bin` + `data/val.bin` | `data/meta.json` (`META_BIN`) |

!!! note
    For binary mode, if `data.processed_dir` points to `data/processed`, `train.py` automatically checks the parent directory (`data/`) for `train.bin` and `val.bin`.

## Checkpointing and Resume Behavior

Produced during training:

- `checkpoints/ckpt_last.pt` (`CHECKPOINT`)
- `checkpoints/ckpt_best.pt` (when `save_best = true`)
- `checkpoints/train_log.json`

!!! warning
    Native resume-from-checkpoint is not implemented in `train.py` yet. `ckpt_last.pt` is for inference/export compatibility and state inspection, not automatic continuation.

## Common Errors

- Binary shards missing: see [Binary shards not found](troubleshooting.md#binary-shards-not-found).
- Metadata path mismatch: see [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Vocab inference failure: see [Char vocab missing](troubleshooting.md#char-vocab-missing).
- CUDA fallback warning: see [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Next / Related

- [Inference & Demo](inference-and-demo.md)
- [Export & Deployment](export-and-deployment.md)
- [Troubleshooting](troubleshooting.md)
