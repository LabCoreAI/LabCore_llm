# Entrainement

L'entrainement est pilote par config TOML avec overrides CLI.

## Commande

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 2000
```

Flags:

- `--config`
- `--max-iters`
- `--device`
- `--tokenizer`

## Presets disponibles

- `configs/base.toml`
- `configs/bpe_medium/bpe_medium_50M.toml`
- `configs/bpe_rope_flash/bpe_50M_rope_flash.toml`
- Variantes de taille disponibles dans `configs/bpe_medium/` et `configs/bpe_rope_flash/` (`5M` a `50M`).

## Comportement runtime

- charge config et metadata
- charge tokenizer
- construit dataloaders selon `training.data_format`
- applique warmup + cosine decay LR
- sauvegarde checkpoints + logs

## Fichiers produits

- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

Le checkpoint inclut:

- config model
- poids model
- etat optimizer
- config trainer
- pertes recentes
- metadata tokenizer

## Conseils perf

- CPU: preset `base.toml` + peu d'iterations
- GPU 8GB: utiliser presets 50M
- ajuster `gradient_accumulation_steps` selon VRAM

## Erreurs connues

### `Binary shards not found`

Regenerer dataset en `bin`.

### `Unable to infer vocab_size`

Definir `model.vocab_size` ou verifier `meta.json`.

## Suite

Voir [Inference et Demo](inference-and-demo.md).
