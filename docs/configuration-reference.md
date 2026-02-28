# Configuration Reference

`train.py` reads TOML configs and applies defaults through `load_config`.

## File Sections

### `[general]`

- `run_name`: free-form experiment name.
- `seed`: random seed value.
- `tokenizer`: `char` or `bpe`.

### `[data]`

- `dataset`: logical dataset label.
- `processed_dir`: path where prepared artifacts live.

### `[model]`

- `vocab_size`
- `block_size`
- `n_layer`
- `n_head`
- `n_embd`
- `dropout`
- `bias`
- `use_rope`
- `use_flash`

### `[optimizer]`

- `learning_rate`
- `weight_decay`
- `beta1`
- `beta2`
- `grad_clip`

### `[training]`

- `batch_size`
- `gradient_accumulation_steps`
- `max_iters`
- `warmup_iters`
- `lr_decay_iters`
- `min_lr`
- `eval_interval`
- `eval_iters`
- `log_interval`
- `save_interval`
- `device`
- `checkpoint_dir`
- `data_format` (`txt` or `bin`)

### `[generation]`

Used mainly for defaults in docs/examples:

- `max_new_tokens`
- `temperature`
- `top_k`

## Built-in Defaults

`load_config` currently ensures:

- `general.tokenizer = "char"` if missing
- `data.processed_dir = "data/processed"` if missing
- `model.use_rope = false` if missing
- `model.use_flash = false` if missing
- `training.data_format = "txt"` if missing

## Example: Minimal Config

```toml
[data]
processed_dir = "data/processed"

[model]
block_size = 128
n_layer = 6
n_head = 8
n_embd = 256
vocab_size = 65

[training]
batch_size = 32
max_iters = 1000
device = "cpu"
```

## Example: 50M RoPE/Flash Style

Use `configs/bpe_50M_rope_flash.toml` as the baseline for binary pipelines.

