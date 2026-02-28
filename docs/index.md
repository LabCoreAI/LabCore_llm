# LabCore LLM

Practical decoder-only GPT framework for local training, evaluation, and deployment.

## Quick Install

```bash
pip install -e ".[torch]"
```

## Quick Example

Tiny Shakespeare with the char tokenizer:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
python train.py --config configs/base.toml --tokenizer char --max-iters 200
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

## What You Get

- Decoder-only GPT training stack with config-driven workflows.
- Optional RoPE positional encoding.
- Optional FlashAttention/SDPA attention path.
- Memory-mapped binary data pipeline (`train.bin`/`val.bin`).
- Hugging Face export workflow.
- GGUF conversion and quantization path.
- LoRA fine-tuning entrypoint for instruction workflows.

## Next Steps

- [Getting Started](getting-started.md)
- [Training](training.md)
- [Export & Deployment](export-and-deployment.md)
