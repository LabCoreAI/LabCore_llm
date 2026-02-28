# LabCore LLM
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Ready-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/GhostPunishR)
[![License](https://img.shields.io/github/license/LabCoreAI/LabCore_llm?style=flat&logo=gnu&logoColor=white&color=2EA44F)](LICENSE)

LabCore LLM is a modular GPT training stack built for practical local workflows:
data preparation, training, generation, web demo, Hugging Face export, and GGUF conversion.

---

## What This Project Delivers

| Area | Included |
|---|---|
| Tokenization | Character tokenizer + GPT-2 compatible BPE (`tiktoken`) |
| Model | Decoder-only GPT with optional RoPE + FlashAttention path |
| Data pipeline | `txt/npy` and `bin/meta.json` pipelines, with mmap loader for binary |
| Training | Config-driven training loop with warmup/cosine LR schedule and checkpointing |
| Inference | CLI generation + Gradio demo (local checkpoint or HF model) |
| Packaging | Editable install, tests, CI workflow, changelog, docs |
| Distribution | Hugging Face export (`safetensors`) + GGUF quantization script |
| Fine-tuning | Instruction LoRA script for HF-compatible CausalLM models |

---

## Installation

### 1) Install PyTorch (CPU or CUDA)

Install PyTorch first using the official selector for your platform:

https://pytorch.org/get-started/locally/

Choose CPU-only or CUDA depending on your hardware.

---

### 2) Install LabCore LLM

Minimal install:

```bash
pip install -e .
```

Install with PyTorch dependency included:

```bash
pip install -e ".[torch]"
```

Developer tools (tests + lint):

```bash
pip install -e ".[torch,dev]"
```

Optional extras:

```bash
pip install -e ".[hf,demo]"
pip install -e ".[gguf]"
pip install -e ".[finetune]"
```

Install everything:

```bash
pip install -e ".[all]"
```

---

## Quick Start

### 1) Char Baseline (small and fast to validate pipeline)

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt
python train.py --config configs/base.toml --tokenizer char --max-iters 200
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

### 2) BPE 50M Baseline

```bash
python scripts/prepare_data.py --dataset wikitext --tokenizer bpe --output-format txt
python train.py --config configs/bpe_medium_50M.toml --tokenizer bpe
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer bpe --prompt "In the beginning"
```

### 3) BPE 50M + RoPE + FlashAttention + Binary Pipeline

```bash
python scripts/prepare_data.py --dataset wikitext --tokenizer bpe --output-format bin --output-dir data
python train.py --config configs/bpe_50M_rope_flash.toml --tokenizer bpe
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/meta.json --tokenizer bpe --prompt "In the beginning"
```

Note: when `--output-format bin --output-dir data` is used, metadata is written to `data/meta.json`.

---

## Config Presets

| Config | Params | Tokenizer | Positional Encoding | Attention Path | Data Format |
|---|---:|---|---|---|---|
| `configs/base.toml` | ~5M | Char | Learned absolute | Standard causal | txt/npy |
| `configs/bpe_medium_50M.toml` | ~50M | BPE (GPT-2) | Learned absolute | Standard causal | txt/npy |
| `configs/bpe_50M_rope_flash.toml` | ~50M | BPE (GPT-2) | RoPE | SDPA + Flash (if available) | bin/mmap |

---

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

`Makefile` is optional convenience; all workflows run directly with Python commands.

---

## Export and Deployment

### Export to Hugging Face format

```bash
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --output-dir outputs/hf_export
```

For binary pipeline metadata:

```bash
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/meta.json --output-dir outputs/hf_export
```

Optional push:

```bash
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --output-dir outputs/hf_export --push --repo-id GhostPunishR/labcore-llm-50M
```

---

### Gradio demo

Local checkpoint:

```bash
python demo_gradio.py --source local --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json
```

Remote model:

```bash
python demo_gradio.py --source hf --repo-id GhostPunishR/labcore-llm-50M
```

---

### GGUF quantization (llama.cpp)

```bash
python scripts/quantize_gguf.py --hf-dir outputs/hf_export --llama-cpp-dir third_party/llama.cpp --quant-type Q4_K_M
python scripts/quantize_gguf.py --hf-dir outputs/hf_export --llama-cpp-dir third_party/llama.cpp --quant-type Q5_K_M
```

---

## Fine-tuning (Instruction / LoRA)

```bash
python scripts/fine_tune_instruction.py --model-id GhostPunishR/labcore-llm-50M --dataset Open-Orca/OpenOrca --output-dir outputs/lora_openorca
```

---

## Repository Layout

```text
LabCore_llm/
|- configs/                # training presets
|- data/                   # raw + processed artifacts
|- scripts/                # prepare/export/quantize/fine-tune utilities
|- src/labcore_llm/
|  |- model/               # GPT + RoPE/Flash path
|  |- tokenizer/           # Char + BPE tokenizers
|  |- trainer/             # training loop + checkpoint logic
|  |- data/                # dataset loaders (txt + bin/mmap)
|  |- config/              # TOML config loader
|- tests/                  # unit tests
|- train.py                # training entrypoint
|- generate.py             # text generation entrypoint
|- demo_gradio.py          # web demo entrypoint
```

---

## Quality and Validation

- CI workflow runs lint + tests.

Local test command:

```bash
python -m pytest -q
```

---

## Hardware Guidance

- CPU-only: good for smoke tests and short runs.
- GPU (recommended): NVIDIA CUDA for practical training speed.
- 50M presets are tuned for mid-range 8 GB VRAM setups (for example RTX 4060 class).
- If running CPU-only, start with `configs/base.toml` and small `--max-iters`.

---

## License

This project is licensed under GPL-3.0.

---

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

---

## Disclaimer

This project is intended for educational and research purposes.  
It is not optimized for large-scale production deployment.
