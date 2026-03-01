# Training

Utilisez cette page pour lancer, monitorer et checkpoint un entrainement avec des reglages coherents.
Prerequis: dataset et metadata prepares depuis [Data Pipeline](data-pipeline.md).

## Command(s)

Commande de reference:

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
```

Exemples d'override device:

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cpu
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cuda
```

Flags `train.py`:

- `--config`: chemin preset TOML (`CONFIG_EXAMPLE = configs/base.toml`)
- `--max-iters`: override runtime du nombre total d'iterations
- `--device`: `cpu` ou `cuda`
- `--tokenizer`: `char` ou `bpe`

## Precision and Gradient Accumulation

Configurez ces valeurs dans `[training]`:

- `grad_accum_steps`: facteur d'accumulation de gradient (default `1`)
- `precision`: `fp32` (default), `fp16` ou `bf16`

`effective_batch_size = batch_size * grad_accum_steps`

La mixed precision est activee seulement si `device = "cuda"` et `precision != "fp32"`.
Sur CPU, l'entrainement repasse en `fp32`.

Exemple RTX 4060:

```toml
[training]
batch_size = 8
grad_accum_steps = 4
precision = "fp16"
```

### Best Checkpoint & Early Stopping

- `save_best` sauvegarde `checkpoints/ckpt_best.pt` quand la loss de validation s'ameliore d'au moins `early_stopping_min_delta`.
- `early_stopping` est desactive par defaut.
- `early_stopping_patience` compte les evaluations sans amelioration suffisante.
- `early_stopping_min_delta` definit le seuil minimal d'amelioration de `val_loss`.

```toml
[training]
early_stopping = true
early_stopping_patience = 3
early_stopping_min_delta = 0.001
save_best = true
```

## `data_format` and Metadata Mapping

| Mode entrainement | Valeur config | Artifacts attendus | Chemin metadata |
|---|---|---|---|
| Pipeline texte | `training.data_format = "txt"` | `data/processed/train.txt` + `data/processed/val.txt` (ou `.npy`) | `data/processed/meta.json` (`META_TXT`) |
| Pipeline binaire | `training.data_format = "bin"` | `data/train.bin` + `data/val.bin` | `data/meta.json` (`META_BIN`) |

!!! note
    En mode binaire, si `data.processed_dir` pointe vers `data/processed`, `train.py` verifie automatiquement le parent (`data/`) pour `train.bin` et `val.bin`.

## Checkpointing and Resume Behavior

Fichiers produits pendant l'entrainement:

- `checkpoints/ckpt_last.pt` (`CHECKPOINT`)
- `checkpoints/ckpt_best.pt` (si `save_best = true`)
- `checkpoints/train_log.json`

!!! warning
    La reprise native depuis checkpoint n'est pas encore implementee dans `train.py`. `ckpt_last.pt` sert surtout a l'inference/export et a l'inspection d'etat.

## Common Errors

- Binaries manquants: voir [Binary shards not found](troubleshooting.md#binary-shards-not-found).
- Mauvais chemin metadata: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Echec d'inference vocab: voir [Char vocab missing](troubleshooting.md#char-vocab-missing).
- Warning CUDA fallback: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Next / Related

- [Inference & Demo](inference-and-demo.md)
- [Export & Deployment](export-and-deployment.md)
- [Troubleshooting](troubleshooting.md)
