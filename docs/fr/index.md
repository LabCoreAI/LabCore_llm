# LabCore LLM

Cette documentation est le guide opérationnel pour exécuter LabCore de bout en bout: préparation des données, entraînement, inférence et export.
Les pages EN dans `docs/` restent la source de vérité, et les pages FR dans `docs/fr/` sont des miroirs complets.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPLv3-2EA44F?style=flat&logo=gnu&logoColor=white)](https://github.com/LabCoreAI/LabCore_llm/blob/HEAD/LICENSE)

## Preset de référence utilisé dans la documentation

Tous les exemples utilisent cette base:

- Dataset: `tinyshakespeare`
- Tokenizer: `char`
- `CONFIG_EXAMPLE = configs/base.toml`
- Override d'entraînement standard: `--max-iters 5000`
- `CHECKPOINT = checkpoints/ckpt_last.pt`
- `META_TXT = data/processed/meta.json`
- `META_BIN = data/meta.json`

!!! tip
    Gardez ces valeurs pour votre premier run complet. La plupart des erreurs viennent d'un mauvais alignement checkpoint/métadonnées.

## Installation rapide

```bash
python -m pip install -e ".[torch,dev]"
```

Pour l'export Hugging Face et l'interface Gradio:

```bash
python -m pip install -e ".[torch,hf,demo]"
```

## Commandes de démarrage rapide

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

Artefacts attendus:

- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`
- `data/processed/meta.json`

## Flux de bout en bout

```text
prepare_data.py -> train.py -> generate.py/demo_gradio.py -> export_hf.py -> quantize_gguf.py
```

## Plan de documentation

### Guides

- [Démarrage](getting-started.md): setup de l'environnement et premier run reproductible.
- [Pipeline de données](data-pipeline.md): création des données `txt` ou `bin` et des métadonnées.
- [Entraînement](training.md): entraînement, checkpointing et mapping des formats.
- [Inférence et démo](inference-and-demo.md): génération CLI et démo Gradio.
- [Ajustement fin](fine-tuning.md): workflow LoRA instruction tuning.
- [Export et déploiement](export-and-deployment.md): export HF et conversion GGUF.

### Référence

- [Configuration](configuration-reference.md): référence complète des clés TOML.
- [Opérations](operations.md): hygiène opérationnelle, release et checks.
- [Dépannage](troubleshooting.md): correctifs rapides avec ancres.
- [Performances](benchmarks.md): benchmark d'inférence (`tok/s`, pic VRAM) et reporting.

### Développement

- [Guide développeur](developer-guide.md): workflow contributeur et commandes de validation.

## Suite / liens

- [Démarrage](getting-started.md)
- [Pipeline de données](data-pipeline.md)
- [Entraînement](training.md)
