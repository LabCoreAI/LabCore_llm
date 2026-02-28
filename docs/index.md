# LabCore LLM

Practical decoder-only GPT framework for local training, evaluation, and deployment.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Ready-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/GhostPunishR)
[![License](https://img.shields.io/github/license/LabCoreAI/LabCore_llm?style=flat&logo=gnu&logoColor=white&color=2EA44F)](https://github.com/LabCoreAI/LabCore_llm/blob/master/LICENSE)

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
