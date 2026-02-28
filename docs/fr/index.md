# LabCore LLM

Framework GPT decoder-only pragmatique pour l'entraînement, l'inférence et le déploiement en local.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/LabCoreAI/LabCore_llm?style=flat&logo=gnu&logoColor=white&color=2EA44F)](https://github.com/LabCoreAI/LabCore_llm/blob/master/LICENSE)

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

## Flux de bout en bout

```text
Corpus texte brut
  -> scripts/prepare_data.py
  -> train.py (+ preset config)
  -> checkpoints/ckpt_last.pt
  -> generate.py / demo_gradio.py
  -> scripts/export_hf.py
  -> scripts/quantize_gguf.py (optionnel)
  -> scripts/fine_tune_instruction.py (optionnel)
```

## Plan de la documentation

### Guides

- [Getting Started](getting-started.md): configuration de l'environnement et premier run.
- [Data Pipeline](data-pipeline.md): génération de datasets `txt` ou `bin` + metadata.
- [Training](training.md): entraînement avec presets et suivi des checkpoints.
- [Inference & Demo](inference-and-demo.md): génération CLI et interface Gradio.
- [Fine-Tuning](fine-tuning.md): workflow LoRA pour l'instruction tuning.
- [Export & Deployment](export-and-deployment.md): export Hugging Face et chemin GGUF.

### Référence

- [Configuration](configuration-reference.md): sections TOML, defaults et exemples.
- [Operations](operations.md): procédures release/sécurité/support.
- [Troubleshooting](troubleshooting.md): erreurs fréquentes et correctifs rapides.
- [Benchmarks](benchmarks.md): modèle de suivi des performances.

### Développement

- [Developer Guide](developer-guide.md): workflow dev local, tests et standards.

## Prochaines étapes rapides

- [Getting Started](getting-started.md)
- [Training](training.md)
- [Export & Deployment](export-and-deployment.md)
- [Operations](operations.md) pour checks qualité, guidance hardware, license et disclaimer.
