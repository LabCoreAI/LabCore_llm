# Operations

Utilisez cette page pour l'hygiene des dossiers operationnels, les checks de validation et les workflows release/securite.
Prerequis: dependances du projet installees dans l'environnement actif.

## Artifact Directories

- `data/`: datasets prepares, incluant `data/meta.json` pour runs `bin`.
- `checkpoints/`: sorties entrainement (`ckpt_last.pt`, `train_log.json`, `ckpt_best.pt` si active).
- `outputs/`: artifacts exportes (`hf_export/`, `gguf/`, sorties fine-tuning).
- `runs/`: logs d'experience optionnels ou exports trackers externes (pas auto-cree par les scripts core).

!!! note
    Les scripts core ecrivent surtout dans `checkpoints/` et `outputs/`. Gardez ces dossiers dans vos notes de run, mais ne committez pas de gros artifacts.

## Command(s)

Checks qualite locaux:

```bash
python -m pytest -q
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

Checks build docs:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build
```

## Output Files / Artifacts Produced

- Logs tests/lint dans le terminal
- Site docs dans `site/` apres `mkdocs build`
- Pipelines CI: `.github/workflows/ci.yml` et `.github/workflows/docs.yml`

## Security and Release

Signalement securite (voir `SECURITY.md`):

- Utiliser GitHub Security Advisories pour les vulnerabilites
- Inclure impact, composants affectes et etapes de reproduction

Flux release (voir `RELEASE.md`):

1. Verifier que la CI est verte.
2. Lancer les validations locales.
3. Mettre a jour la version et le changelog.
4. Tagger et publier la release.

## Common Errors

- Dependances manquantes: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- Confusion metadata/path: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- CUDA attendu mais indisponible: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Related

- [Troubleshooting](troubleshooting.md)
- [Benchmarks](benchmarks.md)
- [Developer Guide](developer-guide.md)
