# Operations

This page centralizes release flow, security reporting, and runtime troubleshooting.

## Security

Do not report vulnerabilities in public issues.
Follow the process in `SECURITY.md`:

- Prefer GitHub Security Advisories
- Share impact, affected components, and reproducible steps

## Release Process

Reference: `RELEASE.md`

1. Ensure CI is green.
2. Run local tests.
3. Bump version in `pyproject.toml`.
4. Update `CHANGELOG.md`.
5. Tag release and push tag.
6. Publish GitHub Release.

## Support Channels

Reference: `SUPPORT.md`

- Usage questions: Discussion/Issue with `support` label
- Bugs: issue template
- Features: feature request template

## Troubleshooting Matrix

### `ModuleNotFoundError: torch`

Install:

```bash
pip install -e ".[torch,dev]"
```

### `Char tokenizer requires vocab in meta.json`

Regenerate data with char tokenizer and use matching metadata path.

### `Binary shards not found`

Prepare binary dataset:

```bash
python scripts/prepare_data.py --dataset wikitext --tokenizer bpe --output-format bin --output-dir data
```

### CUDA fallback warning

If CUDA is unavailable, scripts fall back to CPU automatically.

## Docs Operations

Build docs locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Deploy docs via GitHub Actions:

- Push to `master` or `main`
- `docs.yml` builds and deploys `site/` to `gh-pages`

