# Developer Guide

Cette page cible les contributeurs qui travaillent sur le code du projet.

## Repository Map

```text
src/labcore_llm/
  config/      # loader TOML et defaults
  data/        # abstractions dataset
  model/       # implementation GPT
  tokenizer/   # tokenizers char + BPE
  trainer/     # boucle training, scheduler, checkpointing

scripts/       # helpers data prep, export, quantize, fine-tune
configs/       # presets TOML
tests/         # tests unitaires
```

## Local Dev Environment

```bash
python -m venv .venv
## PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[torch,dev]"
```

## Validation Commands

Lancer les tests:

```bash
python -m pytest -q
```

Lancer le lint aligne sur la CI:

```bash
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

## CI Workflows

- `.github/workflows/ci.yml`: lint + tests
- `.github/workflows/docs.yml`: build et deploiement MkDocs

## Contribution Quality Bar

- Gardez les commits focalises et atomiques.
- Mettez a jour la doc lors des changements de comportement/CLI.
- Ajoutez des tests pour corrections de bugs et nouvelles logiques.
- Ne commitez pas de gros artifacts data/model.

## Packaging Notes

- Le projet utilise un layout `src/` avec setuptools.
- Les groupes de dependances optionnelles sont dans `pyproject.toml`.
- Les scripts d'entree sont des fichiers Python, pas des wrappers console-script.
