# Guide développeur

Cette page cible les contributeurs qui travaillent sur le code du projet.

## Structure du dépôt

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

## Environnement de développement local

```bash
python -m venv .venv
## PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[torch,dev]"
```

## Commandes de validation

Lancer les tests:

```bash
python -m pytest -q
```

Lancer le lint aligné sur la CI:

```bash
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

## Workflows CI

- `.github/workflows/ci.yml`: lint + tests
- `.github/workflows/docs.yml`: build et déploiement MkDocs

## Niveau de qualité des contributions

- Gardez les commits focalisés et atomiques.
- Mettez à jour la documentation lors des changements de comportement/CLI.
- Ajoutez des tests pour les corrections de bugs et les nouvelles logiques.
- Ne commitez pas de gros artefacts de données/modèles.

## Notes de packaging

- Le projet utilise une structure `src/` avec setuptools.
- Les groupes de dépendances optionnelles sont dans `pyproject.toml`.
- Les scripts d'entrée sont des fichiers Python, pas des wrappers console-script.
