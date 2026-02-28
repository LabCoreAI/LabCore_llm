# Accueil

Cette page FR resume le flux principal. La documentation EN (`docs/`) reste la reference complete.
Preset de reference: `tinyshakespeare` + `char` + `configs/base.toml`, checkpoint `checkpoints/ckpt_last.pt`.

## Commandes Rapides

```bash
python -m pip install -e ".[torch,dev]"
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

## Voir aussi

- [Getting Started (EN)](../getting-started.md)
- [Training (EN)](../training.md)
- [Export & Deployment (EN)](../export-and-deployment.md)
