# Guide Dev

Page dediee aux contributeurs code.

## Structure repo

```text
src/labcore_llm/
  config/
  data/
  model/
  tokenizer/
  trainer/

scripts/
configs/
tests/
```

## Environment local

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[torch,dev]"
```

## Validation locale

Tests:

```bash
python -m pytest -q
```

Lint:

```bash
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

## CI

- `.github/workflows/ci.yml`
- `.github/workflows/docs.yml`

## Regles contribution

- commits atomiques
- docs mises a jour en meme temps que le code
- tests ajoutes quand logique modifiee
- pas d'artifacts lourds dans git

