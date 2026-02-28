# Export et Deploiement

Cette page FR resume la sortie HF puis la conversion GGUF.
Point de depart: `checkpoints/ckpt_last.pt` + metadata correspondante.

## Commandes Rapides

```bash
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --output-dir outputs/hf_export
python scripts/export_hf.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --output-dir outputs/hf_export --push --repo-id GhostPunishR/labcore-llm-50M
python scripts/quantize_gguf.py --hf-dir outputs/hf_export --llama-cpp-dir third_party/llama.cpp --output-dir outputs/gguf --quant-type Q4_K_M
```

## Artifacts

- `outputs/hf_export/model.safetensors`
- `outputs/hf_export/config.json`
- `outputs/hf_export/tokenizer.json`
- `outputs/gguf/labcore-50m-f16.gguf`
- `outputs/gguf/labcore-50m-q4_k_m.gguf`

## Voir aussi

- [Export & Deployment (EN)](../export-and-deployment.md)
- [Fine-Tuning (EN)](../fine-tuning.md)
- [Troubleshooting (EN)](../troubleshooting.md)
