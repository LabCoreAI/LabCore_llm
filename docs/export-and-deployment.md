# Export and Deployment

Use this page to package a local checkpoint for Hugging Face and optional GGUF deployment.
Prerequisites: a valid checkpoint (`checkpoints/ckpt_last.pt`) and matching metadata (`data/processed/meta.json` or `data/meta.json`).

## HF Export vs GGUF Export

- HF export creates standard model artifacts for Hugging Face workflows.
- GGUF conversion creates quantized files for `llama.cpp`-style runtimes.

## Command(s)

Export local checkpoint to HF layout:

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export
```

Push exported folder to HF Hub:

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export \
  --push \
  --repo-id GhostPunishR/labcore-llm-50M
```

Convert HF export to GGUF and quantize:

```bash
python scripts/quantize_gguf.py \
  --hf-dir outputs/hf_export \
  --llama-cpp-dir third_party/llama.cpp \
  --output-dir outputs/gguf \
  --quant-type Q4_K_M
```

## Output Files / Artifacts Produced

`outputs/hf_export/`:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `README.md`

`outputs/gguf/`:

- `labcore-50m-f16.gguf`
- `labcore-50m-q4_k_m.gguf` (or `q5_k_m` / both when `--quant-type all`)

!!! warning
    GGUF conversion requires a valid `llama.cpp` checkout with conversion script and quantizer binary available.

## Common Errors

- Metadata mismatch (`txt` vs `bin`): see [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Missing `huggingface_hub`/`safetensors`: see [Torch not installed](troubleshooting.md#torch-not-installed).
- Missing llama.cpp tools: see [Windows path and policy issues](troubleshooting.md#windows-path-policy).

## Next / Related

- [Fine-Tuning](fine-tuning.md)
- [Operations](operations.md)
- [Troubleshooting](troubleshooting.md)
