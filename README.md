# LabCore LLM

LabCore LLM is a modular decoder-only GPT stack focused on clear, reproducible engineering workflows: data prep, training, inference, demo, export, and optional fine-tuning.

[![CI](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/ci.yml)
[![Docs](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/docs.yml/badge.svg)](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/docs.yml)
[![Security](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/security.yml/badge.svg)](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/security.yml)
[![Release](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/release.yml/badge.svg)](https://github.com/LabCoreAI/LabCore_llm/actions/workflows/release.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPLv3-2EA44F?style=flat&logo=gnu&logoColor=white)](LICENSE)

## Core Capabilities

- Decoder-only GPT with optional RoPE + Flash/SDPA path.
- Inference KV-cache for incremental decoding.
- Streaming Gradio demo with minimal multi-turn chat and `system_prompt`.
- Reproducible inference benchmark (`tok/s` + CUDA VRAM peak) with JSON + Markdown output.
- Training with mixed precision (AMP) and gradient accumulation.
- Best checkpoint saving (`ckpt_best.pt`) and optional early stopping.
- Advanced sampling controls (`top_k`, `top_p`, `repetition_penalty`).
- Reproducible generation controls (`seed`, `deterministic`).

## Quick Sanity Run (tinyshakespeare + char)

Install:

```bash
python -m pip install -e ".[torch,dev]"
```

GPU check (optional):

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Prepare data, train 200 iters, generate:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 200
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --config configs/base.toml --tokenizer char --prompt "ROMEO:"
```

## Documentation

- [English Docs](https://labcoreai.github.io/LabCore_llm/)
- [Documentation Francaise](https://labcoreai.github.io/LabCore_llm/fr/)

## Repository Layout

```text
LabCore_llm/
|- configs/
|- docs/
|- scripts/
|- src/labcore_llm/
|- tests/
|- train.py
|- generate.py
|- demo_gradio.py
```

## License

GPL-3.0-only  
Copyright (C) 2026 LabCoreAI

## Disclaimer

This project is intended for educational and research purposes.
It is not optimized for large-scale production deployment.