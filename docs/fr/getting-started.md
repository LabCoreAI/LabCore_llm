# Démarrage

Utilisez cette page pour un premier run reproductible en environ 5 minutes et vérifier que l'environnement est sain.
Prérequis: Python `3.11+`, `pip` et GPU CUDA optionnel.

## Commandes

Installer les dépendances:

```bash
python -m pip install -e ".[torch,dev]"
```

Vérifier CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Démarrage rapide (preset de référence: tinyshakespeare + char + `configs/base.toml`):

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

!!! tip
    Pour un smoke test strict de 5 minutes, lancez l'entraînement, attendez la première évaluation/checkpoint, puis stoppez et lancez la génération.

## Fichiers de sortie / artefacts produits

- `data/processed/meta.json`
- `data/processed/train.txt`, `data/processed/val.txt`
- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

## Checklist de succès

- Les logs affichent au moins deux lignes `train_loss` et une tendance à la baisse.
- Un checkpoint existe dans `checkpoints/ckpt_last.pt`.
- `generate.py` retourne un texte non vide.

## Erreurs fréquentes

- `ModuleNotFoundError: torch`: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- CUDA attendu mais indisponible: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).
- Mauvais chemin de métadonnées: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).

!!! warning
    Gardez `--checkpoint` et `--meta` alignés sur le même run. Mélanger des fichiers de runs différents donne des résultats trompeurs.

## Suite / liens

- [Pipeline de données](data-pipeline.md)
- [Entraînement](training.md)
- [Dépannage](troubleshooting.md)
