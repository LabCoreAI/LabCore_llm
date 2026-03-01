# Référence de configuration

Utilisez cette page comme référence unique des clés TOML utilisées par `train.py` et des valeurs par défaut.
Prérequis: connaître le preset de référence (`configs/base.toml`).

## Valeurs canoniques utilisées dans les guides

- `CONFIG_EXAMPLE = configs/base.toml`
- `CHECKPOINT = checkpoints/ckpt_last.pt`
- `META_TXT = data/processed/meta.json`
- `META_BIN = data/meta.json`

## Référence des sections et des clés

### `[general]`

| Clé | Type | Valeur typique | Notes |
|---|---|---|---|
| `run_name` | string | `"tiny_char_baseline"` | Label informatif du run. |
| `seed` | int | `1337` | `seed` (graine d'initialisation) optionnelle. |
| `tokenizer` | string | `"char"` ou `"bpe"` | Valeur par défaut: `"char"` si omis. |

### `[data]`

| Clé | Type | Valeur typique | Notes |
|---|---|---|---|
| `dataset` | string | `"tinyshakespeare"` | Label des métadonnées. |
| `processed_dir` | string path | `"data/processed"` | Valeur par défaut: `"data/processed"` si omis. |

### `[model]`

| Clé | Type | Valeur typique | Notes |
|---|---|---|---|
| `vocab_size` | int | `65` (char) / `50257` (bpe) | Peut être inféré via metadata/tokenizer si omis. |
| `block_size` | int | `512` | Longueur de séquence. |
| `n_layer` | int | `6` | Profondeur du Transformer. |
| `n_head` | int | `8` | Nombre de têtes d'attention. |
| `n_embd` | int | `256` | Taille des embeddings. |
| `dropout` | float | `0.1` | Taux de dropout. |
| `bias` | bool | `true` | Active/désactive le biais linéaire. |
| `use_rope` | bool | `false`/`true` | Valeur par défaut: `false`. |
| `use_flash` | bool | `false`/`true` | Valeur par défaut: `false`. |

### `[optimizer]`

| Clé | Type | Valeur typique | Notes |
|---|---|---|---|
| `learning_rate` | float | `3e-4` | Fallback vers `[training]` si absent. |
| `weight_decay` | float | `0.01` | Fallback vers `[training]` si absent. |
| `beta1` | float | `0.9` | AdamW beta1. |
| `beta2` | float | `0.95` | AdamW beta2. |
| `grad_clip` | float | `1.0` | Fallback vers `[training]` si absent. |

### `[training]`

| Clé | Type | Valeur typique | Notes |
|---|---|---|---|
| `batch_size` | int | `8` | Taille de batch (micro-batch) par step. |
| `gradient_accumulation_steps` | int | `1` à `8` | Accumulation de gradients pour augmenter la taille de batch effective. |
| `max_iters` | int | `5000` | Valeur de référence de la documentation. |
| `warmup_iters` | int | `0` à `500` | Steps de warmup du learning rate. |
| `lr_decay_iters` | int | `5000` | Souvent aligné sur `max_iters`. |
| `min_lr` | float | `3e-5` | Plancher du learning rate (cosine decay). |
| `eval_interval` | int | `200` | Cadence d'évaluation/checkpoint. |
| `eval_iters` | int | `50` | Nombre de batches de validation par évaluation. |
| `log_interval` | int | `20` | Cadence des logs console. |
| `save_interval` | int | `200` ou `500` | Cadence de sauvegarde supplémentaire. |
| `device` | string | `"cuda"` ou `"cpu"` | Surcharge possible via CLI `--device`. |
| `checkpoint_dir` | string path | `"checkpoints"` | Produit `ckpt_last.pt` et `train_log.json`. |
| `data_format` | string | `"txt"` ou `"bin"` | Valeur par défaut: `"txt"` si omis. |
| `early_stopping` | bool | `false` | Active l'arrêt anticipé sur la loss de validation. |
| `early_stopping_patience` | int | `5` | Nombre d'évaluations sans amélioration avant arrêt. |
| `early_stopping_min_delta` | float | `0.0` | Gain minimal de loss de validation pour compter une amélioration. |
| `save_best` | bool | `true` | Sauvegarde `ckpt_best.pt` sur nouvelle meilleure loss de validation. |

### `[generation]`

| Clé | Type | Valeur typique | Notes |
|---|---|---|---|
| `max_new_tokens` | int | `200` | Valeur pratique pour les scripts docs. |
| `temperature` | float | `0.6` à `0.9` | Contrôle de l'aléatoire du sampling. |
| `top_k` | int | `40` à `50` | Troncature du sampling. |
| `top_p` | float | `0.9` à `1.0` | Seuil nucleus sampling. |
| `repetition_penalty` | float | `1.0` à `1.2` | Pénalité des tokens répétés. |
| `use_kv_cache` | bool | `true` | Active le KV-cache en inférence. |
| `stream` | bool | `true` | Active le streaming token-by-token en démo. |
| `system_prompt` | string | `""` | Prompt système du chat minimal. |
| `max_history_turns` | int | `6` | Nombre max de tours d'historique chat. |
| `seed` | int or null | `1337` | `seed` (graine d'initialisation) pour la reproductibilité. |
| `deterministic` | bool | `false` | Active les algorithmes déterministes PyTorch. |

## Valeurs par défaut appliquées automatiquement

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

## Exemple: configuration minimale

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

## Exemple: point de départ RTX 4060 (8GB, à valider localement)

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
    Le bloc RTX 4060 est un point de départ, pas un profil benchmark garanti. Ajustez selon votre VRAM et votre stack driver.

## Liens

- [Entraînement](training.md)
- [Opérations](operations.md)
- [Performances](benchmarks.md)
