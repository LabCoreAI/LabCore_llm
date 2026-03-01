# Benchmarks

Utilisez le script de benchmark inference pour mesurer de maniere reproductible les `tokens/s` et le pic VRAM.
Le script supporte les checkpoints locaux et les repos Hugging Face, avec export JSON + Markdown.

## Inference Benchmark Script

```bash
python scripts/benchmark_infer.py --help
```

Le runtime par defaut reste court (warmup + 3 runs mesures).

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

- Generation de warmup (`--warmup-tokens`, non comptee dans le throughput final).
- Generation mesuree (`--gen-tokens`) repetee `--iters` fois.
- Resume throughput: `mean`, `min`, `max` tokens/sec.
- VRAM peak (`torch.cuda.max_memory_allocated`) en execution CUDA.

Les reglages de reproductibilite sont lus dans `[generation]` quand `--config` est fourni:

- `seed`
- `deterministic`
- reglages sampling (`temperature`, `top_k`, `top_p`, `repetition_penalty`)
- `use_kv_cache` (sauf override via flags CLI)

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

Collez la ligne Markdown generee (via `--md-out` ou sortie terminal) dans ce tableau.
Ajoutez le JSON dans la description de PR si disponible.

| Device | Source | Model size (params M) | KV-cache | gen_tokens | mean tok/s | peak VRAM MiB |
|---|---|---:|---|---:|---:|---:|
| _your result_ | _local/hf_ | _0.000_ | _on/off_ | _256_ | _0.00_ | _N/A or value_ |
