# Pipeline Data

LabCore gere deux formats de sortie:

- `txt` pour flux texte + numpy
- `bin` pour flux binaire memmap

## Commande Principale

```bash
python scripts/prepare_data.py \
  --dataset tinyshakespeare \
  --tokenizer char \
  --output-format txt \
  --raw-dir data/raw \
  --output-dir data/processed \
  --val-ratio 0.1
```

## Options CLI

- `--dataset`: `tinyshakespeare` ou `wikitext`
- `--tokenizer`: `char` ou `bpe`
- `--output-format`: `txt` ou `bin`
- `--raw-dir`: cache corpus brut
- `--output-dir`: dossier de sortie
- `--val-ratio`: ratio validation

## Arborescence de sortie

### Format `txt`

- `train.txt`
- `val.txt`
- `corpus.txt`
- `train.npy`
- `val.npy`
- `meta.json`

### Format `bin`

Si `output-dir` finit par `processed`, les binaires partent dans le parent:

- `data/train.bin`
- `data/val.bin`
- `data/meta.json`

## Champs `meta.json`

- `dataset`
- `vocab_size`
- `tokenizer`
- `dtype`
- `output_format`
- champs de fichiers texte ou bin selon format

## Notes tokenizer

- `char`: vocab appris sur tout le corpus.
- `bpe`: encodeur GPT-2 via `tiktoken`.

## Bonnes pratiques

- verifier les compteurs de tokens affiches
- verifier presence de `meta.json`
- aligner `training.data_format` avec les artifacts generes

## Suite

Voir [Entrainement](training.md).

