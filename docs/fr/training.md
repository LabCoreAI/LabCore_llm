# Entrainement

Cette page FR resume le lancement d'entrainement et les chemins a respecter.
Le preset de reference est `configs/base.toml` avec `--max-iters 5000`.

## Commandes Rapides

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cpu
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cuda
```

## Artifacts

- checkpoint: `checkpoints/ckpt_last.pt`
- log: `checkpoints/train_log.json`
- metadata `txt`: `data/processed/meta.json`
- metadata `bin`: `data/meta.json`

## Voir aussi

- [Training (EN)](../training.md)
- [Inference & Demo (EN)](../inference-and-demo.md)
- [Troubleshooting (EN)](../troubleshooting.md)
