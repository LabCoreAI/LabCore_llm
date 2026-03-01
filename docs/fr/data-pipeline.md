# Pipeline de données

Utilisez cette page pour préparer les données et les métadonnées avec une structure de sortie prévisible.
Prérequis: dépendances installées depuis [Démarrage](getting-started.md).

## Commandes

Pipeline `txt` de référence (utilisé par `configs/base.toml`):

```bash
python scripts/prepare_data.py \
  --dataset tinyshakespeare \
  --tokenizer char \
  --output-format txt \
  --raw-dir data/raw \
  --output-dir data/processed \
  --val-ratio 0.1
```

Pipeline `bin` alternatif:

```bash
python scripts/prepare_data.py \
  --dataset tinyshakespeare \
  --tokenizer char \
  --output-format bin \
  --raw-dir data/raw \
  --output-dir data/processed \
  --val-ratio 0.1
```

## Fichiers de sortie / artefacts produits

Format `txt` (`output-dir = data/processed`):

- `data/processed/train.txt`
- `data/processed/val.txt`
- `data/processed/corpus.txt`
- `data/processed/train.npy`
- `data/processed/val.npy`
- `data/processed/meta.json` (`META_TXT`)

Format `bin`:

- `data/train.bin`
- `data/val.bin`
- `data/meta.json` (`META_BIN`)

!!! note
    Avec `--output-format bin`, si `--output-dir` se termine par `processed`, les fichiers binaires sont écrits dans le parent (`data/`).

## Sélection du format

- Utilisez `txt` avec `training.data_format = "txt"` et la métadonnée `data/processed/meta.json`.
- Utilisez `bin` avec `training.data_format = "bin"` et la métadonnée `data/meta.json`.

## Erreurs fréquentes

- Binaires manquants: voir [Binary shards not found](troubleshooting.md#binary-shards-not-found).
- Mauvais chemin de métadonnées: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Problème de vocabulaire char: voir [Char vocab missing](troubleshooting.md#char-vocab-missing).

## Suite / liens

- [Entraînement](training.md)
- [Inférence et démo](inference-and-demo.md)
- [Dépannage](troubleshooting.md)
