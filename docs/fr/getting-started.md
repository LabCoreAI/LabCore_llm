# Demarrage

Cette page couvre le setup de base et le premier run fonctionnel.

## Prerequis

- Python `3.11+`
- `pip`
- GPU CUDA optionnel (recommande pour entrainement reel)

Installer PyTorch via le selecteur officiel:

<https://pytorch.org/get-started/locally/>

## Installation

Minimal:

```bash
pip install -e .
```

Recommande pour dev + training:

```bash
pip install -e ".[torch,dev]"
```

Extras:

```bash
pip install -e ".[torch,hf,demo]"
pip install -e ".[gguf]"
pip install -e ".[torch,hf,finetune]"
```

## Smoke Test Rapide

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
python train.py --config configs/base.toml --tokenizer char --max-iters 200
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

Artifacts attendus:

- `data/processed/meta.json`
- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

## Notes Device

- `--device cuda` bascule auto sur CPU si CUDA indisponible.
- Pour CPU: commencer avec `configs/base.toml`.

## Erreurs frequentes

### `No module named 'torch'`

```bash
pip install -e ".[torch,dev]"
```

### Probleme download dataset

`prepare_data.py` utilise `datasets` pour charger les jeux HF.  
Pour `wikitext`, installer:

```bash
pip install -e ".[hf]"
```

`tinyshakespeare` garde un fallback HTTP si `datasets` est indisponible.

## Suite

Voir [Pipeline Data](data-pipeline.md).
