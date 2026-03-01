# Inférence et démo

Utilisez cette page pour lancer une génération CLI déterministe et la démo Gradio locale.
Prérequis: checkpoint entraîné (`checkpoints/ckpt_last.pt`) et métadonnées correspondantes.

## Commandes

Génération CLI (chemins de référence):

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

Démo Gradio depuis checkpoint local:

```bash
python demo_gradio.py \
  --source local \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --device cpu \
  --port 7860
```

Démo Gradio depuis Hugging Face:

```bash
python demo_gradio.py \
  --source hf \
  --repo-id GhostPunishR/labcore-llm-50M \
  --device cpu \
  --port 7860
```

## Réglages de génération stables (mode debug)

Utilisez un sampling conservateur pour déboguer la reproductibilité:

- `temperature = 0.2` pour réduire l'aléatoire
- `top-k = 20` (ou moins)
- `max-new-tokens = 80` pour des checks rapides

!!! tip
    Si la qualité chute brutalement, vérifiez d'abord que `--meta` correspond au même run tokenizer/checkpoint.

### Contrôles de sampling

- `temperature`: met à l'échelle les logits avant sampling.
- `top_k`: conserve uniquement les `k` tokens les plus probables.
- `top_p`: seuil nucleus sampling (`1.0` le désactive).
- `repetition_penalty`: pénalise les tokens déjà générés (`1.0` le désactive).
- `use_kv_cache`: active le KV-cache pendant le décodage.
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

### Génération reproductible

- `seed` (graine d'initialisation): fige les RNG Python/NumPy/Torch pour un sampling reproductible.
- `deterministic`: active les algorithmes déterministes PyTorch (`warn_only=True`) et les réglages cuDNN déterministes.

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

### KV-cache, streaming et chat

- Le KV-cache accélère la génération en réutilisant les key/value attention passées, au lieu de recalculer tout le contexte.
- Le streaming met à jour la sortie de démo de manière incrémentale, token par token.
- Le chat multi-tour construit un prompt avec des marqueurs texte simples:
  - `<|system|>`
  - `<|user|>`
  - `<|assistant|>`
- `max_history_turns` conserve seulement les tours les plus récents pour borner la longueur du prompt.

```toml
[generation]
use_kv_cache = true
stream = true
system_prompt = "You are LabCore LLM."
max_history_turns = 6
```

## Fichiers de sortie / artefacts produits

- CLI: texte généré dans le terminal
- Démo: texte généré dans l'interface Gradio
- Aucun nouveau fichier modèle, sauf export explicite

## Erreurs fréquentes

- Métadonnées tokenizer char manquantes: voir [Char vocab missing](troubleshooting.md#char-vocab-missing).
- Mauvais mapping des métadonnées (`txt` vs `bin`): voir [Meta path mismatch](troubleshooting.md#meta-path-mismatch).
- Fallback CUDA: voir [CUDA not detected](troubleshooting.md#cuda-not-detected).

## Suite / liens

- [Export et déploiement](export-and-deployment.md)
- [Ajustement fin](fine-tuning.md)
- [Dépannage](troubleshooting.md)
