# Inference et Demo

Cette page FR couvre la generation CLI et la demo Gradio avec les chemins standard.
Utilisez les memes metadata et checkpoint que pour l'entrainement.

## Commandes Rapides

```bash
python generate.py --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --tokenizer char --prompt "To be"
python demo_gradio.py --source local --checkpoint checkpoints/ckpt_last.pt --meta data/processed/meta.json --device cpu --port 7860
python demo_gradio.py --source hf --repo-id GhostPunishR/labcore-llm-50M --device cpu --port 7860
```

## Reglages stables (debug)

- `temperature = 0.2`
- `top-k = 20`
- `max-new-tokens = 80`

## Voir aussi

- [Inference & Demo (EN)](../inference-and-demo.md)
- [Export & Deployment (EN)](../export-and-deployment.md)
- [Troubleshooting (EN)](../troubleshooting.md)
