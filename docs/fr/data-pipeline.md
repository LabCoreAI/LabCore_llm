# Data Pipeline

Utilisez cette page pour preparer les donnees et metadata avec une structure de sortie previsible.
Prerequis: dependances installees depuis [Getting Started](getting-started.md).

## Command(s)

Pipeline `txt` de reference (utilise par `configs/base.toml`):

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

## Output Files / Artifacts Produced

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
    Avec `--output-format bin`, si `--output-dir` se termine par `processed`, les fichiers binaires sont ecrits dans le parent (`data/`).

## Format Selection

- Utilisez `txt` avec `training.data_format = "txt"` et metadata `data/processed/meta.json`.
- Utilisez `bin` avec `training.data_format = "bin"` et metadata `data/meta.json`.

## Common Errors

- Binaries manquants: voir [Binary shards not found](troubleshooting.md#binary-shards-not-found).
- Mauvais chemin metadata: voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Probleme de vocab char: voir [Char vocab missing](troubleshooting.md#char-vocab-missing).

## Next / Related

- [Training](training.md)
- [Inference & Demo](inference-and-demo.md)
- [Troubleshooting](troubleshooting.md)
