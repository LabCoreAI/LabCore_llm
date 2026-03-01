# LabCore LLM

This documentation is the operational guide for running LabCore end to end: data preparation, training, inference, and export.
English pages in `docs/` are the source of truth, and French pages in `docs/fr/` are concise mirrors.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPLv3-2EA44F?style=flat&logo=gnu&logoColor=white)](https://github.com/LabCoreAI/LabCore_llm/blob/HEAD/LICENSE)

## Reference Preset Used Across Docs

All examples are standardized on this reference setup:

- Dataset: `tinyshakespeare`
- Tokenizer: `char`
- `CONFIG_EXAMPLE = configs/base.toml`
- Canonical training override: `--max-iters 5000`
- `CHECKPOINT = checkpoints/ckpt_last.pt`
- `META_TXT = data/processed/meta.json`
- `META_BIN = data/meta.json`

!!! tip
    Keep these values unchanged for your first full run. Most failures come from checkpoint and metadata path mismatches.

## Quick Install

```bash
python -m pip install -e ".[torch,dev]"
```

For Hugging Face export and demo UI:

```bash
python -m pip install -e ".[torch,hf,demo]"
```

## Quick Start Commands

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
python train.py --config configs/base.toml --tokenizer char --max-iters 5000
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
```

Expected artifacts:

- `checkpoints/ckpt_last.pt`
- `checkpoints/train_log.json`
- `data/processed/meta.json`

## End-to-End Flow

```text
prepare_data.py -> train.py -> generate.py/demo_gradio.py -> export_hf.py -> quantize_gguf.py
```

## Documentation Map

### Guides

- [Getting Started](getting-started.md): environment setup and a reproducible first run.
- [Data Pipeline](data-pipeline.md): build `txt` or `bin` datasets and metadata.
- [Training](training.md): run training, checkpointing, and format selection.
- [Inference & Demo](inference-and-demo.md): CLI generation and Gradio demo.
- [Fine-Tuning](fine-tuning.md): LoRA instruction tuning workflow.
- [Export & Deployment](export-and-deployment.md): HF export and GGUF conversion.

### Reference

- [Configuration](configuration-reference.md): complete TOML key reference.
- [Operations](operations.md): artifacts, release flow, and operational checks.
- [Troubleshooting](troubleshooting.md): anchored fixes for common failures.
- [Benchmarks](benchmarks.md): benchmark template and reporting method.

### Development

- [Developer Guide](developer-guide.md): contributor workflow and validation commands.

## Next / Related

- [Getting Started](getting-started.md)
- [Data Pipeline](data-pipeline.md)
- [Training](training.md)
