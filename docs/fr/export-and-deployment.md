# Export and Deployment

Utilisez cette page pour packager un checkpoint local vers Hugging Face puis, optionnellement, vers GGUF.
Prerequis: checkpoint valide (`checkpoints/ckpt_last.pt`) et metadata correspondante (`data/processed/meta.json` ou `data/meta.json`).

## HF Export vs GGUF Export

- L'export HF cree des artifacts standards pour les workflows Hugging Face.
- La conversion GGUF cree des fichiers quantifies pour runtimes `llama.cpp`.

## Command(s)

Exporter un checkpoint local au format HF:

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export
```

Pousser le dossier exporte vers HF Hub:

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export \
  --push \
  --repo-id GhostPunishR/labcore-llm-50M
```

Convertir l'export HF en GGUF et quantifier:

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
- `labcore-50m-q4_k_m.gguf` (ou `q5_k_m` / les deux avec `--quant-type all`)

!!! warning
    La conversion GGUF exige un checkout `llama.cpp` valide avec script de conversion et binaire de quantization disponibles.

## Common Errors

- Mauvais metadata mapping (`txt` vs `bin`): voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Dependances `huggingface_hub`/`safetensors` manquantes: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- Outils llama.cpp manquants: voir [Windows path and policy issues](troubleshooting.md#windows-path-policy).

## Next / Related

- [Fine-Tuning](fine-tuning.md)
- [Operations](operations.md)
- [Troubleshooting](troubleshooting.md)
