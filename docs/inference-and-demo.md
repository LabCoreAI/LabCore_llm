# Inference and Demo

LabCore supports two interactive inference paths:

- CLI generation with `generate.py`
- Gradio web UI with `demo_gradio.py`

## CLI Generation

```bash
python generate.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --tokenizer char \
  --prompt "To be" \
  --max-new-tokens 200 \
  --temperature 0.9 \
  --top-k 40 \
  --device cpu
```

Key flags:

- `--checkpoint`
- `--meta`
- `--config` (optional tokenizer fallback)
- `--tokenizer` (`char` or `bpe`)
- `--prompt`
- `--max-new-tokens`
- `--temperature`
- `--top-k`
- `--device`

## Gradio Demo (Local Checkpoint)

```bash
python demo_gradio.py \
  --source local \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --device cpu \
  --port 7860
```

## Gradio Demo (HF Hub)

```bash
python demo_gradio.py \
  --source hf \
  --repo-id GhostPunishR/labcore-llm-50M \
  --device cpu \
  --port 7860
```

## Sampling Behavior

`demo_gradio.py` supports:

- temperature scaling
- top-k filtering
- top-p filtering

`generate.py` supports:

- temperature
- top-k

## Troubleshooting

### Missing tokenizer metadata for char runs

`generate.py` requires char vocab in `meta.json`.
Regenerate data with char tokenizer:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
```

### HF model loading dependencies

For remote demo, install:

```bash
pip install -e ".[hf,demo]"
```

## Next Step

Continue with [Export and Deployment](export-and-deployment.md).

