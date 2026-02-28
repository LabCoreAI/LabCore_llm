# LabCore LLM

LabCore LLM is a modular decoder-only GPT framework focused on practical local workflows:
data preparation, training, inference, demo serving, Hugging Face export, GGUF conversion,
and LoRA fine-tuning.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Ready-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/GhostPunishR)
[![License](https://img.shields.io/github/license/LabCoreAI/LabCore_llm?style=flat&logo=gnu&logoColor=white&color=2EA44F)](LICENSE)

## Overview

| Domain | What is included |
|---|---|
| Tokenization | Character tokenizer + GPT-2-compatible BPE tokenizer (`tiktoken`) |
| Model | Decoder-only GPT with optional RoPE and Flash/SDPA path |
| Training | TOML-driven trainer with warmup + cosine LR, eval loop, checkpoints |
| Data | `txt/npy` pipeline and `bin` pipeline with mmap dataset loader |
| Inference | CLI generation (`generate.py`) and Gradio UI (`demo_gradio.py`) |
| Distribution | Hugging Face export (`safetensors`) + optional GGUF quantization |
| Fine-tuning | LoRA script for HF-compatible CausalLM models |

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

## Installation

### Prerequisites

- Python `3.11+`
- `pip`
- Optional CUDA GPU for practical training speed

Install PyTorch first with the official selector (CPU or CUDA):
<https://pytorch.org/get-started/locally/>

### Dependency Profiles

| Profile | Command | Use case |
|---|---|---|
| Core package | `pip install -e .` | Package + base deps (`numpy`, `tiktoken`) |
| Training runtime | `pip install -e ".[torch]"` | Required to run train/generate/demo |
| Dev workflow | `pip install -e ".[torch,dev]"` | Tests + lint |
| HF + demo workflow | `pip install -e ".[torch,hf,demo]"` | Datasets + HF Hub + Gradio |
| Fine-tuning workflow | `pip install -e ".[torch,hf,finetune]"` | LoRA script + dataset loading |
| Full stack | `pip install -e ".[all]"` | All optional features |

## Quick Start

### 1) Char Pipeline (smoke test)

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
python train.py --config configs/base.toml --tokenizer char --max-iters 200
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

Expected artifacts:
- `data/processed/meta.json`
- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`

### 2) BPE 50M (standard attention, txt pipeline)

```bash
python scripts/prepare_data.py --dataset wikitext --tokenizer bpe --output-format txt
python train.py --config configs/bpe_medium/bpe_medium_50M.toml --tokenizer bpe
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer bpe --prompt "In the beginning"
```

### 3) BPE 50M (RoPE + Flash/SDPA, bin pipeline)

```bash
python scripts/prepare_data.py --dataset wikitext --tokenizer bpe --output-format bin --output-dir data
python train.py --config configs/bpe_rope_flash/bpe_50M_rope_flash.toml --tokenizer bpe
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/meta.json --tokenizer bpe --prompt "In the beginning"
```

Note: with `--output-format bin --output-dir data`, metadata is written to `data/meta.json`.

## Configuration Presets

| Family | Path | Size range | Tokenizer | Positional encoding | Attention | Data format |
|---|---|---|---|---|---|---|
| Base char preset | `configs/base.toml` | ~5M | Char | Learned absolute | Standard causal | `txt/npy` |
| BPE standard presets | `configs/bpe_medium/` | `5M` -> `50M` | BPE | Learned absolute | Standard causal | `txt/npy` |
| BPE RoPE/Flash presets | `configs/bpe_rope_flash/` | `5M` -> `50M` | BPE | RoPE | SDPA + Flash if available | `bin/mmap` |

Common entry presets:
- `configs/bpe_medium/bpe_medium_50M.toml`
- `configs/bpe_rope_flash/bpe_50M_rope_flash.toml`

## Main Commands

| Command | Purpose |
|---|---|
| `python scripts/prepare_data.py ...` | Build dataset artifacts + `meta.json` |
| `python train.py --config ...` | Train a model and save `checkpoints/ckpt_last.pt` |
| `python generate.py ...` | Generate text from a local checkpoint |
| `python demo_gradio.py ...` | Launch Gradio inference UI |
| `python scripts/export_hf.py ...` | Export checkpoint to HF-friendly folder |
| `python scripts/quantize_gguf.py ...` | Convert and quantize exported model to GGUF |
| `python scripts/fine_tune_instruction.py ...` | Run LoRA instruction fine-tuning |

`Makefile` is optional convenience. All workflows also run directly with Python commands.

## Inference and Demo

Local checkpoint demo:

```bash
python demo_gradio.py --source local --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --device cpu --port 7860
```

HF-hosted demo:

```bash
python demo_gradio.py --source hf --repo-id GhostPunishR/labcore-llm-50M --device cpu --port 7860
```

## Export and Deployment

Export checkpoint to HF folder:

```bash
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --output-dir outputs/hf_export
```

Binary pipeline metadata variant:

```bash
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/meta.json --output-dir outputs/hf_export
```

Optional Hub push:

```bash
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --output-dir outputs/hf_export --push --repo-id GhostPunishR/labcore-llm-50M
```

GGUF conversion/quantization (requires `llama.cpp`):

```bash
python scripts/quantize_gguf.py --hf-dir outputs/hf_export --llama-cpp-dir third_party/llama.cpp --quant-type Q4_K_M
python scripts/quantize_gguf.py --hf-dir outputs/hf_export --llama-cpp-dir third_party/llama.cpp --quant-type Q5_K_M
```

## Fine-Tuning (Instruction / LoRA)

```bash
python scripts/fine_tune_instruction.py --model-id GhostPunishR/labcore-llm-50M --dataset Open-Orca/OpenOrca --output-dir outputs/lora_openorca --config configs/bpe_rope_flash/bpe_50M_rope_flash.toml
```

Recommended environment for this command:

```bash
pip install -e ".[torch,hf,finetune]"
```

## Repository Structure

```text
LabCore_llm/
|- configs/                  # training presets (base + BPE families)
|- data/                     # raw + processed artifacts
|- docs/                     # bilingual MkDocs documentation
|- scripts/                  # prepare/export/quantize/fine-tune utilities
|- src/labcore_llm/
|  |- config/                # TOML config loader
|  |- data/                  # dataset loaders (txt + bin/mmap)
|  |- model/                 # GPT + RoPE/Flash path
|  |- tokenizer/             # Char + BPE tokenizers
|  |- trainer/               # training loop + checkpoint logic
|- tests/
|- train.py
|- generate.py
|- demo_gradio.py
```

## Documentation

MkDocs documentation is available in English and French.

Local docs preview:

```bash
pip install -r docs/requirements.txt
python -m mkdocs serve
```

## Quality and Validation

CI workflows:
- `.github/workflows/ci.yml` (lint + tests)
- `.github/workflows/docs.yml` (docs build/deploy)

Local checks:

```bash
python -m pytest -q
ruff check src scripts tests train.py generate.py demo_gradio.py
```

## Hardware Guidance

- CPU-only: suitable for smoke tests and short runs.
- GPU recommended: NVIDIA CUDA for practical training speed.
- `50M` presets are tuned for mid-range ~8 GB VRAM setups (for example RTX 4060 class).
- On CPU, start with `configs/base.toml` and low `--max-iters`.

## License

This project is licensed under GPL-3.0.

## Citation

```bibtex
@software{labcore_llm_2026,
  title = {LabCore LLM},
  author = {GhostPunishR and contributors},
  year = {2026},
  url = {https://github.com/LabCoreAI/LabCore_llm},
  license = {GPL-3.0}
}
```

## Disclaimer

This project is intended for educational and research purposes.
It is not optimized for large-scale production deployment.
