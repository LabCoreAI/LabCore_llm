# LabCore LLM

Cette documentation est le guide operationnel pour executer LabCore de bout en bout: preparation des donnees, entrainement, inference et export.
Les pages EN dans `docs/` restent la source de verite, et les pages FR dans `docs/fr/` sont des miroirs complets.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPLv3-2EA44F?style=flat&logo=gnu&logoColor=white)](https://github.com/LabCoreAI/LabCore_llm/blob/HEAD/LICENSE)

## Preset de reference utilise dans la documentation

Tous les exemples utilisent cette base:

- Dataset: `tinyshakespeare`
- Tokenizer: `char`
- `CONFIG_EXAMPLE = configs/base.toml`
- Override entrainement standard: `--max-iters 5000`
- `CHECKPOINT = checkpoints/ckpt_last.pt`
- `META_TXT = data/processed/meta.json`
- `META_BIN = data/meta.json`

!!! tip
    Gardez ces valeurs pour votre premier run complet. La plupart des erreurs viennent d'un mauvais alignement checkpoint/metadata.

## Installation rapide

```bash
python -m pip install -e ".[torch,dev]"
```

Pour export Hugging Face et interface Gradio:

```bash
python -m pip install -e ".[torch,hf,demo]"
```

## Commandes de demarrage rapide

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

Artifacts attendus:

- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`
- `data/processed/meta.json`

## Flux de bout en bout

```text
prepare_data.py -> train.py -> generate.py/demo_gradio.py -> export_hf.py -> quantize_gguf.py
```

## Plan de documentation

### Guides

- [Getting Started](getting-started.md): setup environnement et premier run reproductible.
- [Data Pipeline](data-pipeline.md): creation des donnees `txt` ou `bin` et metadata.
- [Training](training.md): entrainement, checkpointing et mapping des formats.
- [Inference & Demo](inference-and-demo.md): generation CLI et demo Gradio.
- [Fine-Tuning](fine-tuning.md): workflow LoRA instruction tuning.
- [Export & Deployment](export-and-deployment.md): export HF et conversion GGUF.

### Reference

- [Configuration](configuration-reference.md): reference complete des cles TOML.
- [Operations](operations.md): hygiene operationnelle, release et checks.
- [Troubleshooting](troubleshooting.md): correctifs rapides avec ancres.
- [Benchmarks](benchmarks.md): benchmark inference (`tok/s`, VRAM peak) et reporting.

### Developpement

- [Developer Guide](developer-guide.md): workflow contributeur et commandes de validation.

## Next / Related

- [Getting Started](getting-started.md)
- [Data Pipeline](data-pipeline.md)
- [Training](training.md)
