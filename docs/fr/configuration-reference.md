# Reference Config

`train.py` lit les fichiers TOML puis applique des valeurs par defaut.

## Sections supportees

### `[general]`

- `run_name`
- `seed`
- `tokenizer` (`char` ou `bpe`)

### `[data]`

- `dataset`
- `processed_dir`

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
- `data_format` (`txt` ou `bin`)

### `[generation]`

- `max_new_tokens`
- `temperature`
- `top_k`

## Defauts automatiques

Le loader applique:

- `general.tokenizer = "char"`
- `data.processed_dir = "data/processed"`
- `model.use_rope = false`
- `model.use_flash = false`
- `training.data_format = "txt"`

## Exemple minimal

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

