# Training

Training is driven by TOML configs plus optional CLI overrides.

## Main Command

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 2000
```

CLI flags:

- `--config`: path to TOML preset
- `--max-iters`: override training max iterations
- `--device`: `cpu` or `cuda`
- `--tokenizer`: `char` or `bpe`

## Presets

- `configs/base.toml`: small char baseline (`txt` data format)
- `configs/bpe_medium/bpe_medium_50M.toml`: BPE preset with 50M-class model (`txt`)
- `configs/bpe_rope_flash/bpe_50M_rope_flash.toml`: BPE + RoPE + Flash with `bin` format
- Additional size variants are available in `configs/bpe_medium/` and `configs/bpe_rope_flash/` (`5M` to `50M`).

## Runtime Behavior

- Loads config via `labcore_llm.config.load_config`.
- Resolves `meta.json` from configured processed directory.
- Loads tokenizer from metadata.
- Builds dataset loaders based on `training.data_format`:
  - `txt` or `npy` path
  - `bin` memory-mapped path
- Applies warmup + cosine LR schedule.
- Saves checkpoints and train logs.

## Checkpoints and Logs

Trainer saves:

- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

`ckpt_last.pt` includes:

- model config and model weights
- optimizer state
- trainer config
- latest loss snapshot
- tokenizer metadata

## Performance Guidance

- For CPU smoke tests, use `configs/base.toml` and low `--max-iters`.
- For 8GB VRAM setups, start with provided 50M presets.
- Keep `gradient_accumulation_steps` aligned with memory budget.

## Common Errors

### `Binary shards not found`

You selected `training.data_format = "bin"` without `train.bin` and `val.bin`.
Re-run `prepare_data.py` with `--output-format bin`.

### `Unable to infer vocab_size`

Ensure one source provides vocab size:

- `model.vocab_size` in TOML, or
- `meta.json` contains `vocab_size`, or
- tokenizer metadata is available.

## Next Step

Continue with [Inference and Demo](inference-and-demo.md).
