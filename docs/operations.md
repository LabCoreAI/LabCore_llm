# Operations

Use this page for operational folder hygiene, validation checks, and release/security workflows.
Prerequisite: project dependencies installed in your active environment.

## Artifact Directories

- `data/`: prepared datasets, including `data/meta.json` for `bin` runs.
- `checkpoints/`: training outputs (`ckpt_last.pt`, `train_log.json`).
- `outputs/`: exported artifacts (`hf_export/`, `gguf/`, fine-tuning outputs).
- `runs/`: optional experiment logs or external tracker exports (not auto-created by core scripts).

!!! note
    The core training scripts write to `checkpoints/` and `outputs/`. Keep those directories versioned in your run notes, but do not commit large artifacts.

## Command(s)

Local quality checks:

```bash
python -m pytest -q
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

Docs build checks:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build
```

## Output Files / Artifacts Produced

- Test and lint logs in terminal output
- Built docs in `site/` after `mkdocs build`
- CI pipelines: `.github/workflows/ci.yml` and `.github/workflows/docs.yml`

## Security and Release

Security reporting (see `SECURITY.md`):

- Use GitHub Security Advisories for vulnerabilities
- Include impact, affected components, and reproduction steps

Release flow (see `RELEASE.md`):

1. Confirm CI is green.
2. Run local validation commands.
3. Update version and changelog.
4. Tag and publish the release.

## Common Errors

- Missing dependencies: see [Torch not installed](troubleshooting.md#torch-not-installed).
- Metadata and path confusion: see [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- CUDA expected but unavailable: see [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Related

- [Troubleshooting](troubleshooting.md)
- [Benchmarks](benchmarks.md)
- [Developer Guide](developer-guide.md)
