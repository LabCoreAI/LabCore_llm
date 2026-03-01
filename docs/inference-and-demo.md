# Inference and Demo

Use this page to run deterministic CLI generation and the local Gradio demo.
Prerequisites: a trained checkpoint (`checkpoints/ckpt_last.pt`) and matching metadata.

## Command(s)

CLI generation (reference paths):

```bash
python generate.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --tokenizer char \
  --prompt "To be" \
  --max-new-tokens 200 \
  --temperature 0.9 \
  --top-k 40 \
  --device cpu
```

Gradio demo from local checkpoint:

```bash
python demo_gradio.py \
  --source local \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --device cpu \
  --port 7860
```

Gradio demo from Hugging Face:

```bash
python demo_gradio.py \
  --source hf \
  --repo-id GhostPunishR/labcore-llm-50M \
  --device cpu \
  --port 7860
```

## Stable Generation Settings (Debug Mode)

Use conservative sampling when debugging reproducibility:

- `temperature = 0.2` to reduce randomness
- `top-k = 20` (or lower)
- `max-new-tokens = 80` for quick checks

!!! tip
    If output quality suddenly drops, first verify that `--meta` belongs to the same tokenizer/checkpoint run.

### Sampling Controls

- `temperature`: scales logits before sampling.
- `top_k`: keeps only the `k` most likely tokens.
- `top_p`: nucleus sampling cutoff (`1.0` disables it).
- `repetition_penalty`: penalizes already generated tokens (`1.0` disables it).
- `use_kv_cache`: enables KV cache during decoding.
- `stream`: enables token-by-token output streaming.

`top_p` and `repetition_penalty` are read from `[generation]` when using `generate.py --config ...`.

RTX 4060 stable example:

```toml
[generation]
temperature = 0.6
top_k = 50
top_p = 0.9
repetition_penalty = 1.1
use_kv_cache = true
stream = true
system_prompt = "You are LabCore LLM."
max_history_turns = 6
```

### Reproducible Generation

- `seed`: fixes Python/NumPy/Torch RNGs for repeatable sampling.
- `deterministic`: enables deterministic Torch algorithms (`warn_only=True`) and deterministic cuDNN settings.

```toml
[generation]
temperature = 0.6
top_k = 50
top_p = 0.9
repetition_penalty = 1.1
use_kv_cache = true
stream = true
system_prompt = "You are LabCore LLM."
max_history_turns = 6
seed = 1337
deterministic = true
```

### KV Cache, Streaming, and Chat

- KV cache speeds up generation by reusing past attention keys/values instead of recomputing the whole context.
- Streaming updates the demo output incrementally as each token is sampled.
- Multi-turn chat builds prompts with simple text markers:
  - `<|system|>`
  - `<|user|>`
  - `<|assistant|>`
- `max_history_turns` keeps only the latest turns to bound prompt length.

```toml
[generation]
use_kv_cache = true
stream = true
system_prompt = "You are LabCore LLM."
max_history_turns = 6
```

## Output Files / Artifacts Produced

- CLI: generated text in terminal output
- Demo: generated text in Gradio UI
- No new model files unless you explicitly export

## Common Errors

- Char tokenizer metadata missing: see [Char vocab missing](troubleshooting.md#char-vocab-missing).
- Metadata path mismatch (`txt` vs `bin`): see [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- CUDA fallback behavior: see [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Next / Related

- [Export & Deployment](export-and-deployment.md)
- [Fine-Tuning](fine-tuning.md)
- [Troubleshooting](troubleshooting.md)
