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
mkdocs serve
```

Deploiement:

- push sur `master`/`main`
- workflow docs construit `site/` puis deploy sur `gh-pages`

