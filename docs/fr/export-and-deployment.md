# Export et Deploiement

Cette section couvre l'export HF puis la quantization GGUF.

## Export HF

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export
```

Fichiers generes:

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `README.md`

## Push vers Hugging Face Hub

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export \
  --push \
  --repo-id GhostPunishR/labcore-llm-50M
```

## Conversion GGUF

```bash
python scripts/quantize_gguf.py \
  --hf-dir outputs/hf_export \
  --llama-cpp-dir third_party/llama.cpp \
  --output-dir outputs/gguf \
  --quant-type Q4_K_M
```

Valeurs `--quant-type`:

- `Q4_K_M`
- `Q5_K_M`
- `all`

## Checklist de publication

1. valider generation locale
2. verifier coherence `config.json` + tokenizer
3. versionner artifacts
4. publier model card claire

## Suite

Voir [Fine-Tuning](fine-tuning.md).

