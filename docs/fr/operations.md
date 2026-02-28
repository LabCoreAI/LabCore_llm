# Operations

Page centralisant release, securite et depannage.

## Securite

Ne pas publier les vuln dans les issues publiques.
Suivre `SECURITY.md`:

- GitHub Security Advisories en priorite
- details impact + composants + reproduction

## Process release

Reference: `RELEASE.md`

1. CI verte
2. tests locaux ok
3. bump version dans `pyproject.toml`
4. update `CHANGELOG.md`
5. tag release
6. publier GitHub Release

## Support

Reference: `SUPPORT.md`

- usage: discussion/issue label `support`
- bug: template bug
- feature: template feature request

## Qualite et validation

Checks locaux principaux:

```bash
python -m pytest -q
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

Workflows CI:

- `.github/workflows/ci.yml` (lint + tests)
- `.github/workflows/docs.yml` (build/deploy docs)

## Guidance hardware

- CPU-only: suffisant pour smoke tests et runs courts.
- GPU CUDA recommande pour un entrainement pratique.
- Pour ~8 GB VRAM, commencer avec les familles 50M puis ajuster batch/sequence.

## Depannage rapide

### `No module named 'torch'`

```bash
pip install -e ".[torch,dev]"
```

### `Char tokenizer requires vocab in meta.json`

Regenerer metadata avec tokenizer char.

### `Binary shards not found`

```bash
python scripts/prepare_data.py --dataset wikitext --tokenizer bpe --output-format bin --output-dir data
```

### Warning CUDA fallback

Les scripts basculent auto CPU si CUDA indisponible.

## Operations docs

Build local:

```bash
pip install -r docs/requirements.txt
python -m mkdocs serve
```

Deploiement:

- push sur `master`/`main`
- workflow docs construit `site/` puis deploy sur `gh-pages`

## License

Le projet est sous GPL-3.0.  
Voir [LICENSE](https://github.com/LabCoreAI/LabCore_llm/blob/master/LICENSE).

## Disclaimer

Projet destine a l'education et la recherche.
Non optimise pour un deploiement production a grande echelle.
