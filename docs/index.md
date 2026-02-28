# LabCore LLM

Practical decoder-only GPT framework for local training, evaluation, and deployment.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
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

## End-to-End Flow

```text
Raw text corpus
  -> scripts/prepare_data.py
  -> train.py (+ config preset)
  -> checkpoints/ckpt_last.pt
  -> generate.py / demo_gradio.py
  -> scripts/export_hf.py
  -> scripts/quantize_gguf.py (optional)
  -> scripts/fine_tune_instruction.py (optional)
```

## Documentation Map

### Guides

- [Getting Started](getting-started.md): environment setup and first run.
- [Data Pipeline](data-pipeline.md): build `txt` or `bin` datasets + metadata.
- [Training](training.md): run training presets and monitor checkpoints.
- [Inference & Demo](inference-and-demo.md): CLI generation and Gradio workflows.
- [Fine-Tuning](fine-tuning.md): LoRA instruction-tuning workflow.
- [Export & Deployment](export-and-deployment.md): HF export and GGUF path.

### Reference

- [Configuration](configuration-reference.md): TOML sections, defaults, and examples.
- [Operations](operations.md): release/security/support operations.
- [Troubleshooting](troubleshooting.md): common errors and quick fixes.
- [Benchmarks](benchmarks.md): template for performance tracking.

### Development

- [Developer Guide](developer-guide.md): local dev workflow, tests, and standards.

## Quick Next Steps

- [Getting Started](getting-started.md)
- [Training](training.md)
- [Export & Deployment](export-and-deployment.md)
- [Operations](operations.md) for quality checks, hardware guidance, license, and disclaimer.
