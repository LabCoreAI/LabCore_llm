# LabCore LLM

LabCore LLM is a modular decoder-only GPT framework designed for engineers who want a clear,
reproducible training stack without hidden abstractions.

It covers the full lifecycle of a local LLM workflow:
data preparation, training, inference, web demo serving, Hugging Face export,
GGUF conversion, and optional LoRA fine-tuning.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/LabCoreAI/LabCore_llm?style=flat&logo=gnu&logoColor=white&color=2EA44F)](LICENSE)

---

## Why LabCore LLM

Most frameworks abstract the training pipeline behind large ecosystems.
LabCore keeps the stack explicit and inspectable:

- From raw corpus to checkpoints
- From checkpoints to HF export
- From HF export to GGUF deployment

Everything remains script-driven and reproducible.

---

## Core Capabilities

| Domain | Included |
|---|---|
| Tokenization | Character tokenizer and GPT-2-compatible BPE (`tiktoken`) |
| Model | Decoder-only GPT with optional RoPE and Flash/SDPA path |
| Training | TOML-driven loop, warmup + cosine LR, evaluation cadence, checkpoints |
| Data | `txt/npy` and `bin/mmap` pipelines with explicit metadata contracts |
| Inference | CLI generation and Gradio interface |
| Distribution | Hugging Face export (`safetensors`) + optional GGUF quantization |
| Fine-tuning | LoRA instruction tuning for HF-compatible CausalLM models |

---

## Quick Start

Reference preset: `tinyshakespeare` + `char` + `configs/base.toml`.

```bash
python -m pip install -e ".[torch,dev]"
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

Multilingual preset example (`configs/base-multi.toml`, RTX 4060-class target):

```bash
python -m pip install -e ".[torch,dev,hf]"
python scripts/prepare_data.py --dataset wikitext --tokenizer bpe --output-format bin --output-dir data
python train.py --config configs/base-multi.toml
python generate.py --checkpoint checkpoints_multi/ckpt_last.pt --meta data/meta.json --config configs/base-multi.toml --tokenizer bpe --prompt "Bonjour"
```

Current data-prep scope: `scripts/prepare_data.py` supports `tinyshakespeare` and `wikitext` dataset names.

---

## Preset Families

| Family | Path | Sizes |
|---|---|---|
| Base preset | `configs/base.toml` | baseline char configuration |
| Base multi preset | `configs/base-multi.toml` | multilingual-ready 20M profile for RTX 4060-class GPUs |
| BPE standard | `configs/bpe_medium/` | 5M -> 50M parameter range |
| BPE RoPE/Flash | `configs/bpe_rope_flash/` | 5M -> 50M parameter range |

---

## End-to-End Flow

```text
Raw corpus
  -> scripts/prepare_data.py
  -> train.py (+ config preset)
  -> checkpoints/ckpt_last.pt
  -> generate.py / demo_gradio.py
  -> scripts/export_hf.py
  -> scripts/quantize_gguf.py (optional)
  -> scripts/fine_tune_instruction.py (optional)
```

---

## Documentation

The complete user and operational documentation is available on the MkDocs site:

- English: https://labcoreai.github.io/LabCore_llm/
- French: https://labcoreai.github.io/LabCore_llm/fr/

Recommended entry points:

- [Getting Started](docs/getting-started.md)
- [Data Pipeline](docs/data-pipeline.md)
- [Training](docs/training.md)
- [Inference & Demo](docs/inference-and-demo.md)
- [Export & Deployment](docs/export-and-deployment.md)
- [Operations](docs/operations.md)

---

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

---

## Quality Standards

- CI: `.github/workflows/ci.yml` and `.github/workflows/docs.yml`
- Tests and lint aligned with CI gates
- Documentation-first approach

---

## License

GPL-3.0

---

## Disclaimer

This project is intended for educational and research purposes.
It is not optimized for large-scale production deployment.
