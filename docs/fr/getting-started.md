# Getting Started

Utilisez cette page pour un premier run reproductible en environ 5 minutes et verifier que l'environnement est sain.
Prerequis: Python `3.11+`, `pip`, et GPU CUDA optionnel.

## Command(s)

Installer les dependances:

```bash
python -m pip install -e ".[torch,dev]"
```

Verifier CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Demarrage rapide (preset de reference: tinyshakespeare + char + `configs/base.toml`):

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

!!! tip
    Pour un smoke test strict de 5 minutes, lancez l'entrainement, attendez le premier eval/checkpoint, puis stoppez et lancez la generation.

## Output Files / Artifacts Produced

- `data/processed/meta.json`
- `data/processed/train.txt`, `data/processed/val.txt`
- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

## Success Checklist

- Les logs affichent au moins deux lignes `train_loss` et une tendance a la baisse.
- Un checkpoint existe dans `checkpoints/ckpt_last.pt`.
- `generate.py` retourne un texte non vide.

## Common Errors

- `ModuleNotFoundError: torch`: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- CUDA attendu mais indisponible: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).
- Mauvais metadata path: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).

!!! warning
    Gardez `--checkpoint` et `--meta` alignes sur le meme run. Melanger des fichiers de runs differents donne des resultats trompeurs.

## Next / Related

- [Data Pipeline](data-pipeline.md)
- [Training](training.md)
- [Troubleshooting](troubleshooting.md)
