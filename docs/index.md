# LabCore LLM Documentation

LabCore LLM is a modular GPT training stack designed for practical local workflows:
data preparation, model training, text generation, web demo, HF export, and GGUF conversion.

This documentation is organized to let you go from zero to reproducible runs quickly.

## Project Scope

- Tokenization: character tokenizer and GPT-2 compatible BPE tokenizer (`tiktoken`).
- Modeling: decoder-only GPT with optional RoPE and Flash attention path.
- Training: config-driven trainer with warmup + cosine decay, checkpoints, and eval loop.
- Inference: command line generation and Gradio UI for local or HF-hosted assets.
- Deployment: export to safetensors/HF layout, then optional GGUF quantization.
- Fine-tuning: LoRA script for instruction tuning on HF-compatible CausalLM models.

## Start Here

1. Install dependencies in [Getting Started](getting-started.md).
2. Build a dataset with [Data Pipeline](data-pipeline.md).
3. Train a model with [Training](training.md).
4. Run generation and web UI via [Inference and Demo](inference-and-demo.md).
5. Export and publish with [Export and Deployment](export-and-deployment.md).

## Architecture at a Glance

```text
Raw text -> prepare_data.py -> data/processed or data/*.bin + meta.json
         -> train.py + TOML config -> checkpoints/ckpt_last.pt
         -> generate.py / demo_gradio.py
         -> export_hf.py -> outputs/hf_export
         -> quantize_gguf.py -> outputs/gguf
```

## Core Entrypoints

- `train.py`: main training entrypoint.
- `generate.py`: text generation from local checkpoints.
- `demo_gradio.py`: local/HF interactive demo.
- `scripts/prepare_data.py`: dataset preparation.
- `scripts/export_hf.py`: model export and optional HF push.
- `scripts/quantize_gguf.py`: GGUF conversion and quantization.
- `scripts/fine_tune_instruction.py`: LoRA instruction tuning flow.

## Screenshot

![LabCore Gradio UI](assets/gradio-demo.svg)
