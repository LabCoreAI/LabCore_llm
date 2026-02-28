# Demarrage

Cette page FR donne un demarrage rapide et fiable. Utilisez la page EN pour le detail complet.
Prerequis: Python `3.11+`, `pip`, et idealement un GPU CUDA.

## Commandes Rapides

```bash
python -m pip install -e ".[torch,dev]"
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

## Validation rapide

- `checkpoints/ckpt_last.pt` existe
- `data/processed/meta.json` existe
- la generation retourne du texte non vide

## Voir aussi

- [Getting Started (EN)](../getting-started.md)
- [Data Pipeline (EN)](../data-pipeline.md)
- [Troubleshooting (EN)](../troubleshooting.md)
