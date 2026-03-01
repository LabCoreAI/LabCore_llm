# Export et déploiement

Utilisez cette page pour packager un checkpoint local vers Hugging Face puis, optionnellement, vers GGUF.
Prérequis: checkpoint valide (`checkpoints/ckpt_last.pt`) et métadonnées correspondantes (`data/processed/meta.json` ou `data/meta.json`).

## Export HF vs export GGUF

- L'export HF crée des artefacts standards pour les workflows Hugging Face.
- La conversion GGUF crée des fichiers quantifiés pour les runtimes `llama.cpp`.

## Commandes

Exporter un checkpoint local au format HF:

```bash
python scripts/export_hf.py \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --output-dir outputs/hf_export
```

Pousser le dossier exporté vers HF Hub:

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

## Fichiers de sortie / artefacts produits

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

## Erreurs fréquentes

- Mauvais mapping des métadonnées (`txt` vs `bin`): voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Dépendances `huggingface_hub`/`safetensors` manquantes: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- Outils `llama.cpp` manquants: voir [Windows path and policy issues](troubleshooting.md#windows-path-policy).

## Suite / liens

- [Ajustement fin](fine-tuning.md)
- [Opérations](operations.md)
- [Dépannage](troubleshooting.md)
