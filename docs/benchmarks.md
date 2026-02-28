# Benchmarks

Use this page to track reproducible performance numbers across presets and devices.

## Benchmark Template

| Preset | Device | Batch / Seq / Accum | Throughput (it/s or tok/s) | Max VRAM | Notes |
|---|---|---|---:|---:|---|
| `configs/base.toml` | `cpu` | `32 / 128 / 1` | `0.00` | `0.0 GB` | placeholder |

## Minimal Example Entry

Use one line per run with exact settings:

| Preset | Device | Batch / Seq / Accum | Throughput (it/s or tok/s) | Max VRAM | Notes |
|---|---|---|---:|---:|---|
| `configs/bpe_rope_flash/bpe_30M_rope_flash.toml` | `cuda` | `8 / 512 / 8` | `TBD` | `TBD` | fill after first stable run |

## Benchmark Tips

- Warm up for a few iterations before recording.
- Keep dataset, tokenizer, and precision consistent between runs.
- Log GPU model, driver, and PyTorch version in notes.
