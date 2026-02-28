# LabCore LLM

Framework GPT decoder-only pragmatique pour l'entraînement, l'inférence et le déploiement en local.

## Installation rapide

```bash
pip install -e ".[torch]"
```

## Exemple rapide

Tiny Shakespeare avec tokenizer char:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
python train.py --config configs/base.toml --tokenizer char --max-iters 200
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

## Ce que vous obtenez

- Stack GPT decoder-only pilotée par configuration TOML.
- RoPE optionnel.
- Chemin d'attention FlashAttention/SDPA optionnel.
- Pipeline binaire memory-mapped (`train.bin`/`val.bin`).
- Export Hugging Face.
- Conversion GGUF.
- Point d'entrée LoRA pour l'instruction tuning.

## Prochaines étapes

- [Getting Started](getting-started.md)
- [Training](training.md)
- [Export & Deployment](export-and-deployment.md)
