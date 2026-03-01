# Configuration Reference

Utilisez cette page comme reference unique des cles TOML de `train.py` et des defaults.
Prerequis: connaitre le preset de reference (`configs/base.toml`).

## Canonical Values Used in Guides

- `CONFIG_EXAMPLE = configs/base.toml`
- `CHECKPOINT = checkpoints/ckpt_last.pt`
- `META_TXT = data/processed/meta.json`
- `META_BIN = data/meta.json`

## Section and Key Reference

### `[general]`

| Key | Type | Valeur typique | Notes |
|---|---|---|---|
| `run_name` | string | `"tiny_char_baseline"` | Label informatif de run. |
| `seed` | int | `1337` | Champ seed optionnel. |
| `tokenizer` | string | `"char"` ou `"bpe"` | Default `"char"` si omis. |

### `[data]`

| Key | Type | Valeur typique | Notes |
|---|---|---|---|
| `dataset` | string | `"tinyshakespeare"` | Label metadata. |
| `processed_dir` | string path | `"data/processed"` | Default `"data/processed"` si omis. |

### `[model]`

| Key | Type | Valeur typique | Notes |
|---|---|---|---|
| `vocab_size` | int | `65` (char) / `50257` (bpe) | Peut etre infere via metadata/tokenizer si omis. |
| `block_size` | int | `512` | Longueur de sequence. |
| `n_layer` | int | `6` | Profondeur Transformer. |
| `n_head` | int | `8` | Nombre de tetes attention. |
| `n_embd` | int | `256` | Taille embeddings. |
| `dropout` | float | `0.1` | Taux dropout. |
| `bias` | bool | `true` | Active/desactive le bias lineaire. |
| `use_rope` | bool | `false`/`true` | Default `false`. |
| `use_flash` | bool | `false`/`true` | Default `false`. |

### `[optimizer]`

| Key | Type | Valeur typique | Notes |
|---|---|---|---|
| `learning_rate` | float | `3e-4` | Fallback sur `[training]` si absent. |
| `weight_decay` | float | `0.01` | Fallback sur `[training]` si absent. |
| `beta1` | float | `0.9` | AdamW beta1. |
| `beta2` | float | `0.95` | AdamW beta2. |
| `grad_clip` | float | `1.0` | Fallback sur `[training]` si absent. |

### `[training]`

| Key | Type | Valeur typique | Notes |
|---|---|---|---|
| `batch_size` | int | `8` | Micro-batch size par step. |
| `gradient_accumulation_steps` | int | `1` a `8` | Multiplicateur batch effectif. |
| `max_iters` | int | `5000` | Valeur de reference docs. |
| `warmup_iters` | int | `0` a `500` | Steps de warmup LR. |
| `lr_decay_iters` | int | `5000` | Souvent aligne sur `max_iters`. |
| `min_lr` | float | `3e-5` | Floor cosine LR. |
| `eval_interval` | int | `200` | Cadence eval/checkpoint. |
| `eval_iters` | int | `50` | Batches validation par eval. |
| `log_interval` | int | `20` | Cadence logs console. |
| `save_interval` | int | `200` ou `500` | Cadence de sauvegarde supplementaire. |
| `device` | string | `"cuda"` ou `"cpu"` | Override par CLI `--device`. |
| `checkpoint_dir` | string path | `"checkpoints"` | Produit `ckpt_last.pt` et `train_log.json`. |
| `data_format` | string | `"txt"` ou `"bin"` | Default `"txt"` si omis. |
| `early_stopping` | bool | `false` | Active l'arret anticipe sur loss de validation. |
| `early_stopping_patience` | int | `5` | Nombre d'evaluations sans amelioration avant stop. |
| `early_stopping_min_delta` | float | `0.0` | Gain minimal de loss de validation pour compter une amelioration. |
| `save_best` | bool | `true` | Sauvegarde `ckpt_best.pt` sur nouvelle meilleure loss de validation. |

### `[generation]`

| Key | Type | Valeur typique | Notes |
|---|---|---|---|
| `max_new_tokens` | int | `200` | Default pratique pour scripts docs. |
| `temperature` | float | `0.6` a `0.9` | Randomness sampling. |
| `top_k` | int | `40` a `50` | Troncature sampling. |
| `top_p` | float | `0.9` a `1.0` | Nucleus sampling cutoff. |
| `repetition_penalty` | float | `1.0` a `1.2` | Penalite tokens repetes. |
| `use_kv_cache` | bool | `true` | Active KV-cache en inference. |
| `stream` | bool | `true` | Active streaming token-by-token en demo. |
| `system_prompt` | string | `""` | Prompt system du chat minimal. |
| `max_history_turns` | int | `6` | Nombre max de tours d'historique chat. |
| `seed` | int or null | `1337` | Seed de reproductibilite. |
| `deterministic` | bool | `false` | Active algorithmes deterministes PyTorch. |

## Loader Defaults Applied Automatically

`load_config` garantit:

- `general.tokenizer = "char"`
- `data.processed_dir = "data/processed"`
- `model.use_rope = false`
- `model.use_flash = false`
- `training.data_format = "txt"`
- `training.early_stopping = false`
- `training.early_stopping_patience = 5`
- `training.early_stopping_min_delta = 0.0`
- `training.save_best = true`
- `generation.top_p = 1.0`
- `generation.repetition_penalty = 1.0`
- `generation.use_kv_cache = true`
- `generation.stream = true`
- `generation.system_prompt = ""`
- `generation.max_history_turns = 6`
- `generation.seed = null`
- `generation.deterministic = false`

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
    Le bloc RTX 4060 est un point de depart, pas un profil benchmark garanti. Ajustez selon votre VRAM et stack driver.

## Related

- [Training](training.md)
- [Operations](operations.md)
- [Benchmarks](benchmarks.md)
