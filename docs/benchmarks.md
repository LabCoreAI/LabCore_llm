# Benchmarks

Use this page to log reproducible performance results for the same doc preset and path conventions.
Prerequisite: run with fixed config, tokenizer, and data format before recording numbers.

## Command(s)

Reference benchmark launch command:

```bash
python train.py --config configs/base.toml --tokenizer char --max-iters 5000 --device cuda
```

## Benchmark Table Template

| Preset | Dataset / Tokenizer | Device | Batch / Seq / Accum | Throughput (it/s or tok/s) | Max VRAM | Notes |
|---|---|---|---|---:|---:|---|
| `configs/base.toml` | `tinyshakespeare / char` | `cpu` | `8 / 512 / 1` | `TBD` | `TBD` | template row |

## RTX 4060 Reference Entry

No validated RTX 4060 measurement is checked into this repository yet.
Use this row as a fill-in template for your first stable run:

| Preset | Dataset / Tokenizer | Device | Batch / Seq / Accum | Throughput (it/s or tok/s) | Max VRAM | Notes |
|---|---|---|---|---:|---:|---|
| `configs/base.toml` | `tinyshakespeare / char` | `RTX 4060` | `8 / 512 / 2` | `TBD` | `TBD` | record driver + torch version |

## How to Add Your Benchmark

1. Run at least one warmup phase before measuring.
2. Log exact preset path, overrides, and `training.data_format`.
3. Record hardware (`GPU`, driver) and software (`torch`, CUDA version).
4. Add one table row per run and avoid mixing metrics from different settings.

!!! tip
    If numbers look unstable, start with the [Stable generation and debug guidance](inference-and-demo.md#stable-generation-settings-debug-mode) and verify no fallback to CPU.

## Related

- [Training](training.md)
- [Configuration Reference](configuration-reference.md)
- [Operations](operations.md)
