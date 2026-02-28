# Configuration Reference

Use this page as the single source of truth for `train.py` TOML keys and defaults.
Prerequisite: familiarity with the reference preset (`configs/base.toml`).

## Canonical Values Used in Guides

- `CONFIG_EXAMPLE = configs/base.toml`
- `CHECKPOINT = checkpoints/ckpt_last.pt`
- `META_TXT = data/processed/meta.json`
- `META_BIN = data/meta.json`

## Section and Key Reference

### `[general]`

| Key | Type | Typical value | Notes |
|---|---|---|---|
| `run_name` | string | `"tiny_char_baseline"` | Informational run label. |
| `seed` | int | `1337` | Optional random seed field. |
| `tokenizer` | string | `"char"` or `"bpe"` | Default is `"char"` if omitted. |

### `[data]`

| Key | Type | Typical value | Notes |
|---|---|---|---|
| `dataset` | string | `"tinyshakespeare"` | Metadata label. |
| `processed_dir` | string path | `"data/processed"` | Default is `"data/processed"` if omitted. |

### `[model]`

| Key | Type | Typical value | Notes |
|---|---|---|---|
| `vocab_size` | int | `65` (char) / `50257` (bpe) | Can be inferred from metadata/tokenizer when omitted. |
| `block_size` | int | `512` | Sequence length. |
| `n_layer` | int | `6` | Transformer depth. |
| `n_head` | int | `8` | Attention heads. |
| `n_embd` | int | `256` | Embedding size. |
| `dropout` | float | `0.1` | Dropout rate. |
| `bias` | bool | `true` | Linear layer bias toggle. |
| `use_rope` | bool | `false`/`true` | Default `false`. |
| `use_flash` | bool | `false`/`true` | Default `false`. |

### `[optimizer]`

| Key | Type | Typical value | Notes |
|---|---|---|---|
| `learning_rate` | float | `3e-4` | Falls back to `[training]` if missing. |
| `weight_decay` | float | `0.01` | Falls back to `[training]` if missing. |
| `beta1` | float | `0.9` | AdamW beta1. |
| `beta2` | float | `0.95` | AdamW beta2. |
| `grad_clip` | float | `1.0` | Falls back to `[training]` if missing. |

### `[training]`

| Key | Type | Typical value | Notes |
|---|---|---|---|
| `batch_size` | int | `8` | Micro-batch size per step. |
| `gradient_accumulation_steps` | int | `1` to `8` | Effective batch multiplier. |
| `max_iters` | int | `5000` | Docs reference value. |
| `warmup_iters` | int | `0` to `500` | LR warmup steps. |
| `lr_decay_iters` | int | `5000` | Usually align with `max_iters`. |
| `min_lr` | float | `3e-5` | Cosine LR floor. |
| `eval_interval` | int | `200` | Eval/checkpoint cadence. |
| `eval_iters` | int | `50` | Validation batches per eval. |
| `log_interval` | int | `20` | Console logging cadence. |
| `save_interval` | int | `200` or `500` | Optional extra save cadence. |
| `device` | string | `"cuda"` or `"cpu"` | CLI `--device` overrides this. |
| `checkpoint_dir` | string path | `"checkpoints"` | Produces `ckpt_last.pt` and `train_log.json`. |
| `data_format` | string | `"txt"` or `"bin"` | Default is `"txt"` if omitted. |

### `[generation]`

| Key | Type | Typical value | Notes |
|---|---|---|---|
| `max_new_tokens` | int | `200` | Useful for shared defaults in docs. |
| `temperature` | float | `0.6` to `0.9` | Sampling randomness. |
| `top_k` | int | `40` to `50` | Sampling truncation. |

## Loader Defaults Applied Automatically

`load_config` guarantees:

- `general.tokenizer = "char"`
- `data.processed_dir = "data/processed"`
- `model.use_rope = false`
- `model.use_flash = false`
- `training.data_format = "txt"`

## Example: Minimal Small Config

```toml
[general]
tokenizer = "char"

[data]
dataset = "tinyshakespeare"
processed_dir = "data/processed"

[model]
block_size = 256
n_layer = 4
n_head = 4
n_embd = 192
vocab_size = 65
dropout = 0.1
bias = true
use_rope = false
use_flash = false

[training]
batch_size = 8
gradient_accumulation_steps = 1
max_iters = 2000
eval_interval = 200
eval_iters = 50
log_interval = 20
device = "cpu"
checkpoint_dir = "checkpoints"
data_format = "txt"
```

## Example: RTX 4060 Starting Point (8GB, Validate Locally)

```toml
[general]
tokenizer = "char"

[data]
dataset = "tinyshakespeare"
processed_dir = "data/processed"

[model]
block_size = 512
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1
bias = true
use_rope = false
use_flash = false

[optimizer]
learning_rate = 3e-4
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

[training]
batch_size = 8
gradient_accumulation_steps = 2
max_iters = 5000
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-5
eval_interval = 200
eval_iters = 50
log_interval = 20
save_interval = 200
device = "cuda"
checkpoint_dir = "checkpoints"
data_format = "txt"
```

!!! note
    The RTX 4060 block above is a starting template, not a guaranteed benchmark profile. Adjust based on your exact VRAM and driver stack.

## Related

- [Training](training.md)
- [Operations](operations.md)
- [Benchmarks](benchmarks.md)
