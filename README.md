# LabCore LLM

LabCore LLM is a modular decoder-only GPT framework for practical local AI workflows:
data preparation, training, inference, web demo serving, Hugging Face export, GGUF conversion,
and LoRA fine-tuning.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/LabCoreAI/LabCore_llm?style=flat&logo=gnu&logoColor=white&color=2EA44F)](LICENSE)

## Framework Positioning

LabCore LLM is designed for engineers who want a clear, reproducible GPT stack without hidden platform abstractions.
It keeps the workflow explicit from raw corpus to exported artifacts, while staying flexible across CPU, CUDA, and deployment targets.

## Core Capabilities

| Domain | Included in LabCore LLM |
|---|---|
| Tokenization | Character tokenizer and GPT-2-compatible BPE (`tiktoken`) |
| Model | Decoder-only GPT with optional RoPE and Flash/SDPA path |
| Training | TOML-driven loop, warmup + cosine LR, eval cadence, checkpoints |
| Data | `txt/npy` and `bin/mmap` pipelines with metadata contracts |
| Inference | CLI generation and Gradio interface |
| Distribution | Hugging Face export (`safetensors`) + optional GGUF quantization |
| Fine-tuning | LoRA instruction tuning for HF-compatible CausalLM models |

## Preset Families

| Family | Path | Sizes |
|---|---|---|
| Base preset | `configs/base.toml` | baseline char config |
| BPE standard | `configs/bpe_medium/` | `5M`, `10M`, `15M`, `20M`, `30M`, `35M`, `40M`, `45M`, `50M` |
| BPE RoPE/Flash | `configs/bpe_rope_flash/` | `5M`, `10M`, `15M`, `20M`, `30M`, `35M`, `40M`, `45M`, `50M` |

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

## Documentation (Primary Source)

The complete user and operational documentation lives on the MkDocs site:

- English: <https://labcoreai.github.io/LabCore_llm/>
- French: <https://labcoreai.github.io/LabCore_llm/fr/>

Suggested entry points:

- Getting Started: environment and first run
- Data Pipeline: dataset preparation and `meta.json` behavior
- Training: presets, runtime behavior, checkpoints
- Inference & Demo: local/HF generation paths
- Export & Deployment: HF export and GGUF path
- Operations: quality checks, hardware guidance, license, citation, disclaimer

## Repository Layout

```text
LabCore_llm/
|- configs/                  # training preset families
|- docs/                     # bilingual MkDocs documentation
|- scripts/                  # data prep/export/quantize/fine-tune utilities
|- src/labcore_llm/          # framework source package
|- tests/
|- train.py
|- generate.py
|- demo_gradio.py
```

## Quality

- CI: `.github/workflows/ci.yml` and `.github/workflows/docs.yml`
- Local standards: tests + lint aligned with CI gates
- Documentation-first approach: implementation details and runbooks are maintained on the docs site

## License

This project is licensed under GPL-3.0.

## Disclaimer

This project is intended for educational and research purposes.
It is not optimized for large-scale production deployment.
