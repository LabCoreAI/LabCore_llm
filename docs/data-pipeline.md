# Data Pipeline

LabCore supports two dataset output formats:

- `txt` format for text + numpy training flows
- `bin` format for memory-mapped binary training flows

## Main Command

```bash
python scripts/prepare_data.py \
  --dataset tinyshakespeare \
  --tokenizer char \
  --output-format txt \
  --raw-dir data/raw \
  --output-dir data/processed \
  --val-ratio 0.1
```

## CLI Options

- `--dataset`: `tinyshakespeare` or `wikitext`
- `--tokenizer`: `char` or `bpe`
- `--output-format`: `txt` or `bin`
- `--raw-dir`: cache directory for raw corpus files
- `--output-dir`: artifact output directory
- `--val-ratio`: train/validation split ratio

## Output Layout

### `txt` format (`output-dir = data/processed`)

- `train.txt`
- `val.txt`
- `corpus.txt`
- `train.npy`
- `val.npy`
- `meta.json`

### `bin` format

If output dir ends with `processed`, binaries are written to parent:

- `data/train.bin`
- `data/val.bin`
- `data/meta.json`

Otherwise binaries are written directly in your `--output-dir`.

## `meta.json` Fields

- `dataset`
- `vocab_size`
- `tokenizer`
- `dtype`
- `output_format`
- Optional file fields:
  - `train_text_file`, `val_text_file` for `txt`
  - `train_bin_file`, `val_bin_file` for `bin`

## Tokenizer Notes

- `char`: vocabulary is fitted on full corpus and stored in metadata.
- `bpe`: uses GPT-2 encoding (`tiktoken`) and stores tokenizer type + encoding name.

## Validation Tips

- Confirm token counts printed by `prepare_data.py`.
- Ensure `meta.json` exists before launching `train.py`.
- For `bin` runs, keep `training.data_format = "bin"` in config.

## Next Step

Continue with [Training](training.md).

