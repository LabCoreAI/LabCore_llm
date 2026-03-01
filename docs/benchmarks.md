# Benchmarks

Use the inference benchmark script to measure reproducible `tokens/s` and peak VRAM.
The script supports both local checkpoints and Hugging Face repos and can export JSON + Markdown outputs.

## Inference Benchmark Script

```bash
python scripts/benchmark_infer.py --help
```

Default runtime is intentionally short (warmup + 3 measured runs).

### Local Example

```bash
python scripts/benchmark_infer.py \
  --source local \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --tokenizer char \
  --config configs/base.toml \
  --device cpu \
  --json-out outputs/bench_infer.json \
  --md-out outputs/bench_infer.md
```

### Hugging Face Example

```bash
python scripts/benchmark_infer.py \
  --source hf \
  --repo-id LabCoreAI/<id> \
  --config configs/base.toml \
  --device cuda \
  --json-out outputs/bench_infer_hf.json \
  --md-out outputs/bench_infer_hf.md
```

## What Is Measured

- Warmup generation (`--warmup-tokens`, not counted in final throughput).
- Measured generation (`--gen-tokens`) repeated `--iters` times.
- Throughput summary: `mean`, `min`, `max` tokens/sec.
- Peak VRAM (`torch.cuda.max_memory_allocated`) when CUDA is used.

Reproducibility settings are read from `[generation]` when `--config` is provided:

- `seed`
- `deterministic`
- sampling settings (`temperature`, `top_k`, `top_p`, `repetition_penalty`)
- `use_kv_cache` (unless overridden by CLI flag)

## JSON Output Schema (Summary)

```json
{
  "timestamp": "...",
  "commit": "...",
  "platform": {"os": "...", "python": "..."},
  "torch": {"version": "...", "cuda": "..."},
  "device": {"type": "cpu|cuda", "name": "..."},
  "model": {"source": "local|hf", "params_m": 0.0, "block_size": 0, "n_layer": 0, "n_head": 0, "n_embd": 0},
  "generation": {"prompt": "...", "gen_tokens": 256, "temperature": 0.9, "top_k": 40, "top_p": 1.0, "repetition_penalty": 1.0, "use_kv_cache": true},
  "results": {"iters": 3, "tokens_per_sec": {"mean": 0.0, "min": 0.0, "max": 0.0}, "vram_peak_mib": null}
}
```

## Community Results

Paste the generated Markdown row (from `--md-out` or terminal output) into this table.
Attach the JSON output in the PR description if available.

| Device | Source | Model size (params M) | KV-cache | gen_tokens | mean tok/s | peak VRAM MiB |
|---|---|---:|---|---:|---:|---:|
| _your result_ | _local/hf_ | _0.000_ | _on/off_ | _256_ | _0.00_ | _N/A or value_ |
