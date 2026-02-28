# Documentation LabCore LLM

LabCore LLM est une stack GPT modulaire pour des workflows locaux:
preparation des donnees, entrainement, generation, demo web, export HF et conversion GGUF.

Cette documentation couvre tout le cycle projet, en pratique.

## Ce que le projet apporte

- Tokenization: tokenizer character et tokenizer BPE compatible GPT-2 (`tiktoken`)
- Model: GPT decoder-only avec option RoPE + Flash attention
- Training: boucle pilotee par config TOML avec checkpoints et eval
- Inference: CLI et interface Gradio
- Deployment: export safetensors/HF + quantization GGUF
- Fine-tuning: script LoRA pour jeux instruction

## Parcours recommande

1. [Demarrage](getting-started.md)
2. [Pipeline Data](data-pipeline.md)
3. [Entrainement](training.md)
4. [Inference et Demo](inference-and-demo.md)
5. [Export et Deploiement](export-and-deployment.md)

## Vue architecture

```text
Corpus brut -> prepare_data.py -> data/processed ou data/*.bin + meta.json
            -> train.py + config TOML -> checkpoints/ckpt_last.pt
            -> generate.py / demo_gradio.py
            -> export_hf.py -> outputs/hf_export
            -> quantize_gguf.py -> outputs/gguf
```

## Entrees principales

- `train.py`
- `generate.py`
- `demo_gradio.py`
- `scripts/prepare_data.py`
- `scripts/export_hf.py`
- `scripts/quantize_gguf.py`
- `scripts/fine_tune_instruction.py`

![LabCore Gradio UI](../assets/gradio-demo.svg)

