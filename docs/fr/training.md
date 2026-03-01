# Entraînement

Utilisez cette page pour lancer, monitorer et checkpoint un entraînement avec des réglages cohérents.
Prérequis: dataset et métadonnées préparés depuis [Pipeline de données](data-pipeline.md).

## Commandes

Commande de référence:

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
```

Exemples de surcharge du device:

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cpu
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cuda
```

Flags `train.py`:

- `--config`: chemin du preset TOML (`CONFIG_EXAMPLE = configs/base.toml`)
- `--max-iters`: surcharge runtime du nombre total d'itérations
- `--device`: `cpu` ou `cuda`
- `--tokenizer`: `char` ou `bpe`

## Précision et accumulation de gradients

Configurez ces valeurs dans `[training]`:

- `grad_accum_steps`: facteur d'accumulation de gradients (valeur par défaut `1`)
- `precision`: `fp32` (valeur par défaut), `fp16` ou `bf16`

`effective_batch_size = batch_size * grad_accum_steps`

La précision mixte est activée seulement si `device = "cuda"` et `precision != "fp32"`.
Sur CPU, l'entraînement repasse en `fp32`.

Exemple RTX 4060:

```toml
[training]
batch_size = 8
grad_accum_steps = 4
precision = "fp16"
```

### Meilleur checkpoint et arrêt anticipé

- `save_best` sauvegarde `checkpoints/ckpt_best.pt` quand la loss de validation s'améliore d'au moins `early_stopping_min_delta`.
- `early_stopping` est désactivé par défaut.
- `early_stopping_patience` compte les évaluations sans amélioration suffisante.
- `early_stopping_min_delta` définit le seuil minimal d'amélioration de `val_loss`.

```toml
[training]
early_stopping = true
early_stopping_patience = 3
early_stopping_min_delta = 0.001
save_best = true
```

## `data_format` et mapping des métadonnées

| Mode entraînement | Valeur config | Artefacts attendus | Chemin métadonnées |
|---|---|---|---|
| Pipeline texte | `training.data_format = "txt"` | `data/processed/train.txt` + `data/processed/val.txt` (ou `.npy`) | `data/processed/meta.json` (`META_TXT`) |
| Pipeline binaire | `training.data_format = "bin"` | `data/train.bin` + `data/val.bin` | `data/meta.json` (`META_BIN`) |

!!! note
    En mode binaire, si `data.processed_dir` pointe vers `data/processed`, `train.py` vérifie automatiquement le parent (`data/`) pour `train.bin` et `val.bin`.

## Checkpointing et reprise

Fichiers produits pendant l'entraînement:

- `checkpoints/ckpt_last.pt` (`CHECKPOINT`)
- `checkpoints/ckpt_best.pt` (si `save_best = true`)
- `checkpoints/train_log.json`

!!! warning
    La reprise native depuis checkpoint n'est pas encore implémentée dans `train.py`. `ckpt_last.pt` sert surtout à l'inférence/export et à l'inspection d'état.

## Erreurs fréquentes

- Binaires manquants: voir [Binary shards not found](troubleshooting.md#binary-shards-not-found).
- Mauvais chemin de métadonnées: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Échec d'inférence du vocabulaire: voir [Char vocab missing](troubleshooting.md#char-vocab-missing).
- Avertissement de fallback CUDA: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Suite / liens

- [Inférence et démo](inference-and-demo.md)
- [Export et déploiement](export-and-deployment.md)
- [Dépannage](troubleshooting.md)
