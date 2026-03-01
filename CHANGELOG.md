# Changelog

## [0.2.2] - 2026-03-02

### Added
- Inference runtime was extended with incremental KV-cache support:
  - per-layer `past_key_values` flow in attention and GPT forward path
  - cache truncation (sliding window) aligned with `block_size`
  - generation toggle `use_kv_cache` with backward-compatible fallback path
- Streaming generation support was added:
  - `GPT.generate(..., stream=True)` yields token IDs incrementally
  - non-stream path remains unchanged in behavior and return type
- Gradio demo capabilities were expanded:
  - token-by-token streaming output
  - minimal multi-turn chat prompt building (`<|system|>`, `<|user|>`, `<|assistant|>`)
  - configurable `system_prompt` and bounded history (`max_history_turns`)
  - generation controls in UI (`KV cache`, `stream`, sampling sliders)
- Reproducible inference benchmark tooling was added:
  - new script `scripts/benchmark_infer.py` (local + HF sources)
  - throughput reporting (`tokens/s` mean/min/max), CUDA VRAM peak, JSON + Markdown outputs
  - configurable warmup/iters/token counts for quick runs
- Shared reproducibility utility was introduced:
  - `src/labcore_llm/utils/repro.py` (`configure_generation_reproducibility`)
  - reused by generation and benchmark flows
- Training loop controls were extended:
  - best checkpoint saving (`ckpt_best.pt`) with `save_best`
  - optional early stopping on validation loss
  - `early_stopping_patience` and `early_stopping_min_delta`

### Changed
- Configuration defaults were expanded and validated:
  - `generation.use_kv_cache`, `generation.stream`, `generation.system_prompt`, `generation.max_history_turns`
  - `training.early_stopping`, `training.early_stopping_patience`, `training.early_stopping_min_delta`, `training.save_best`
- Training checkpoint writing was generalized to support both `ckpt_last.pt` and `ckpt_best.pt` with the same payload schema.
- `train.py` and `generate.py` now read and apply the new config controls without introducing new CLI flags.
- MkDocs configuration was aligned with current bilingual docs:
  - explicit `edit_uri`
  - refreshed FR navigation labels and nav consistency

### Fixed
- Generation parity and reproducibility coverage for new inference paths:
  - KV-cache vs non-cache equivalence
  - stream vs non-stream equivalence
  - cache truncation stability
- Training behavior coverage for new stopping/checkpoint logic:
  - best-checkpoint creation
  - early-stopping trigger and default non-trigger behavior
- French documentation parity with English across all guide/reference/development pages.

### Removed
- Legacy short-form FR documentation stubs were replaced by full-structure parity pages.

## [0.2.1] - 2026-03-01

### Added
- Nucleus sampling (`top_p`) and repetition penalty controls were added to generation:
  - model runtime (`labcore_llm.model.gpt`)
  - CLI/config usage (`generate.py`, `configs/base.toml`)
  - automated coverage (`tests/test_generation_sampling.py`)
- Generation reproducibility controls were added:
  - `generation.seed`
  - `generation.deterministic`
  - deterministic setup helper (`configure_generation_reproducibility` in `generate.py`)
- Training improvements were added:
  - gradient accumulation support (`grad_accum_steps`, alias `gradient_accumulation_steps`)
  - mixed precision modes (`fp16`, `bf16`, `fp32`) with autocast and scaler handling in `Trainer`
  - dedicated training tests for accumulation and precision behavior
- Preset coverage was expanded in both model families:
  - `configs/bpe_medium/`: `5M`, `10M`, `15M`, `20M`, `30M`, `35M`, `40M`, `45M`, `50M`
  - `configs/bpe_rope_flash/`: `5M`, `10M`, `15M`, `20M`, `30M`, `35M`, `40M`, `45M`, `50M`
- Documentation platform was expanded:
  - bilingual MkDocs site (EN/FR)
  - grouped navigation (`Guides`, `Reference`, `Development`)
  - dedicated Troubleshooting and Benchmarks pages in both languages

### Changed
- Config loading behavior was tightened:
  - explicit defaults for `generation.top_p`, `generation.repetition_penalty`, `generation.seed`, `generation.deterministic`
  - validation for generation seed and deterministic flags
  - null seed sentinel handling in TOML parsing
- Base configuration (`configs/base.toml`) was updated to reflect new training and generation controls.
- README was repositioned as a high-level project entry page; operational tutorials are now centered in MkDocs.
- Documentation and dependency guidance were aligned with runtime profiles:
  - demo stack: `.[torch,hf,demo]`
  - fine-tuning stack: `.[torch,hf,finetune]`
  - data-prep guidance for datasets dependency (`.[hf]`)
- MkDocs UX was refined for production usage:
  - removed page edit/view actions
  - removed heading permanent-link anchors
  - improved navigation and mobile readability

### Fixed
- HF export reliability with tied GPT weights (`scripts/export_hf.py`, safetensors save path).
- CI test collection reliability by installing torch in CI test environment.
- Test import-path reliability for root-level scripts (`tests/conftest.py` now adds repo root and `src`).
- Base preset regressions corrected (`configs/base.toml`).
- French navigation labels and language switch consistency in docs.
- Heading hierarchy cleanup (`docs/developer-guide.md`) for cleaner table-of-contents behavior.

### Removed
- `CITATION.cff` was removed.
- Citation sections were removed from README and docs operations pages.

## [0.2.0] - 2026-02-27

### Added
- Initial public project baseline shipped across the full repository:
  - decoder-only GPT core (`src/labcore_llm/model`) with char and BPE tokenization paths
  - data pipeline (`scripts/prepare_data.py`) with metadata contract (`meta.json`) and txt/bin outputs
  - training and inference CLIs (`train.py`, `generate.py`)
  - local/HF Gradio demo (`demo_gradio.py`)
  - HF export flow (`scripts/export_hf.py`) and optional GGUF conversion (`scripts/quantize_gguf.py`)
  - optional LoRA instruction fine-tuning entrypoint (`scripts/fine_tune_instruction.py`)
  - package structure and optional dependency profiles in `pyproject.toml`
  - baseline automated testing and CI/docs workflows
