# Export and Deployment

This section covers export to HF-compatible artifacts and optional GGUF quantization.

## Export Checkpoint to HF Layout

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export
```

Generated files:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `README.md`

## Push to Hugging Face Hub

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export \
  --push \
  --repo-id GhostPunishR/labcore-llm-50M
```

Requirements:

- `huggingface_hub`
- valid HF authentication (`huggingface-cli login` or token in env)

## GGUF Conversion and Quantization

Prepare llama.cpp first, then run:

```bash
python scripts/quantize_gguf.py \
  --hf-dir outputs/hf_export \
  --llama-cpp-dir third_party/llama.cpp \
  --output-dir outputs/gguf \
  --quant-type Q4_K_M
```

Supported quant choices:

- `Q4_K_M`
- `Q5_K_M`
- `all`

## Deployment Checklist

1. Verify `generate.py` works on exported artifacts.
2. Keep `config.json` and tokenizer metadata in sync with checkpoint.
3. Version artifacts by model family and date.
4. Publish model card with known limits and intended usage.

## Next Step

Continue with [Fine-Tuning](fine-tuning.md).

