# Inference and Demo

Utilisez cette page pour lancer une generation CLI deterministe et la demo Gradio locale.
Prerequis: checkpoint entraine (`checkpoints/ckpt_last.pt`) et metadata correspondante.

## Command(s)

Generation CLI (chemins de reference):

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

Demo Gradio depuis checkpoint local:

```bash
python demo_gradio.py \
  --source local \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --device cpu \
  --port 7860
```

Demo Gradio depuis Hugging Face:

```bash
python demo_gradio.py \
  --source hf \
  --repo-id GhostPunishR/labcore-llm-50M \
  --device cpu \
  --port 7860
```

## Stable Generation Settings (Debug Mode)

Utilisez un sampling conservateur pour deboguer la reproductibilite:

- `temperature = 0.2` pour reduire l'aleatoire
- `top-k = 20` (ou moins)
- `max-new-tokens = 80` pour des checks rapides

!!! tip
    Si la qualite chute brutalement, verifiez d'abord que `--meta` correspond au meme run tokenizer/checkpoint.

### Sampling Controls

- `temperature`: met a l'echelle les logits avant sampling.
- `top_k`: conserve uniquement les `k` tokens les plus probables.
- `top_p`: cutoff nucleus sampling (`1.0` le desactive).
- `repetition_penalty`: penalise les tokens deja generes (`1.0` le desactive).
- `use_kv_cache`: active le KV-cache pendant le decodage.
- `stream`: active le streaming token-by-token.

`top_p` et `repetition_penalty` sont lus depuis `[generation]` avec `generate.py --config ...`.

Exemple stable RTX 4060:

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

- `seed`: fige les RNG Python/NumPy/Torch pour un sampling reproductible.
- `deterministic`: active les algorithmes deterministes PyTorch (`warn_only=True`) et les reglages cuDNN deterministes.

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

- Le KV-cache accelere la generation en reutilisant les key/value attention passees au lieu de recalculer tout le contexte.
- Le streaming met a jour la sortie demo incrementale token par token.
- Le chat multi-tour construit un prompt avec des marqueurs texte simples:
  - `<|system|>`
  - `<|user|>`
  - `<|assistant|>`
- `max_history_turns` conserve seulement les tours les plus recents pour borner la longueur du prompt.

```toml
[generation]
use_kv_cache = true
stream = true
system_prompt = "You are LabCore LLM."
max_history_turns = 6
```

## Output Files / Artifacts Produced

- CLI: texte genere dans le terminal
- Demo: texte genere dans l'UI Gradio
- Aucun nouveau fichier modele sauf export explicite

## Common Errors

- Metadata tokenizer char manquante: voir [Char vocab missing](troubleshooting.md#char-vocab-missing).
- Mauvais mapping metadata (`txt` vs `bin`): voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Fallback CUDA: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Next / Related

- [Export & Deployment](export-and-deployment.md)
- [Fine-Tuning](fine-tuning.md)
- [Troubleshooting](troubleshooting.md)
