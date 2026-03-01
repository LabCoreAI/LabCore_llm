# Opérations

Utilisez cette page pour l'hygiène des dossiers opérationnels, les checks de validation et les workflows de release/sécurité.
Prérequis: dépendances du projet installées dans l'environnement actif.

## Répertoires d'artefacts

- `data/`: datasets préparés, incluant `data/meta.json` pour les runs `bin`.
- `checkpoints/`: sorties d'entraînement (`ckpt_last.pt`, `train_log.json`, `ckpt_best.pt` si activé).
- `outputs/`: artefacts exportés (`hf_export/`, `gguf/`, sorties de fine-tuning).
- `runs/`: logs d'expérience optionnels ou exports de trackers externes (pas créés automatiquement par les scripts core).

!!! note
    Les scripts core écrivent surtout dans `checkpoints/` et `outputs/`. Gardez ces dossiers dans vos notes de run, mais ne commitez pas de gros artefacts.

## Commandes

Checks qualité locaux:

```bash
python -m pytest -q
ruff check src scripts tests train.py generate.py demo_gradio.py --select E9,F63,F7,F82
```

Checks de build docs:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build
```

## Fichiers de sortie / artefacts produits

- Logs tests/lint dans le terminal
- Site docs dans `site/` après `mkdocs build`
- Pipelines CI: `.github/workflows/ci.yml` et `.github/workflows/docs.yml`

## Sécurité et release

Signalement sécurité (voir `SECURITY.md`):

- Utiliser GitHub Security Advisories pour les vulnérabilités
- Inclure l'impact, les composants affectés et les étapes de reproduction

Flux de release (voir `RELEASE.md`):

1. Vérifier que la CI est verte.
2. Lancer les validations locales.
3. Mettre à jour la version et le changelog.
4. Tagger et publier la release.

## Erreurs fréquentes

- Dépendances manquantes: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- Confusion metadata/path: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- CUDA attendu mais indisponible: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Liens

- [Dépannage](troubleshooting.md)
- [Performances](benchmarks.md)
- [Guide développeur](developer-guide.md)
