# Inference et Demo

Deux modes principaux:

- generation CLI (`generate.py`)
- interface web Gradio (`demo_gradio.py`)

## Generation CLI

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

## Demo Gradio locale

```bash
python demo_gradio.py \
  --source local \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --device cpu \
  --port 7860
```

## Demo Gradio depuis HF

```bash
python demo_gradio.py \
  --source hf \
  --repo-id GhostPunishR/labcore-llm-50M \
  --device cpu \
  --port 7860
```

## Sampling

`demo_gradio.py` supporte:

- temperature
- top-k
- top-p

`generate.py` supporte:

- temperature
- top-k

## Depannage

### tokenizer char sans vocab

Regenerer la data char + `meta.json`:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
```

### dependances HF demo manquantes

```bash
pip install -e ".[hf,demo]"
```

## Suite

Voir [Export et Deploiement](export-and-deployment.md).

