# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning.

## [Unreleased]

### Added
- Structured bilingual documentation site (English/French) with grouped navigation (`Guides`, `Reference`, `Development`).
- Dedicated troubleshooting and benchmark pages in EN/FR (`docs/*` and `docs/fr/*`).
- Additional model preset scales for both preset families:
  - `configs/bpe_medium/`: `5M`, `10M`, `15M`, `20M`, `30M`, `35M`, `40M`, `45M`, `50M`
  - `configs/bpe_rope_flash/`: `5M`, `10M`, `15M`, `20M`, `30M`, `35M`, `40M`, `45M`, `50M`

### Changed
- README refocused as a presentation page; procedural tutorials moved to MkDocs.
- Documentation aligned with runtime behavior and dependency profiles:
  - HF demo profile documented as `.[torch,hf,demo]`
  - fine-tuning profile documented as `.[torch,hf,finetune]`
- `scripts/prepare_data.py` dependency guidance updated to point to `.[hf]` for `datasets`.
- MkDocs UX cleaned for production use:
  - removed page edit/view actions
  - removed heading permanent-link anchors
  - improved navigation structure and mobile readability

### Removed
- `CITATION.cff` removed.
- Citation sections removed from README and docs operations pages.

### Fixed
- French navigation labels and language switch consistency.
- Heading hierarchy cleanup (`docs/developer-guide.md`) for cleaner table of contents behavior.

## [0.2.0] - 2026-02-27

### Added
- Core decoder-only GPT framework with char and BPE tokenization paths.
- Data preparation pipeline for `txt/npy` and `bin` outputs with `meta.json`.
- CLI workflows for training (`train.py`) and generation (`generate.py`).
- Gradio demo for local checkpoints and Hugging Face models (`demo_gradio.py`).
- Export pipeline to Hugging Face format (`scripts/export_hf.py`).
- GGUF conversion/quantization helper (`scripts/quantize_gguf.py`).
- Instruction fine-tuning entrypoint with LoRA (`scripts/fine_tune_instruction.py`).
- Packaging and optional dependency profiles in `pyproject.toml` (`torch`, `dev`, `hf`, `demo`, `gguf`, `finetune`, `all`).
- CI workflows for code quality/tests and docs deployment (`.github/workflows/ci.yml`, `.github/workflows/docs.yml`).
