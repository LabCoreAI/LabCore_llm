# Performances

Utilisez le script de benchmark d'inférence pour mesurer de manière reproductible les `tokens/s` et le pic de VRAM.
Le script prend en charge les checkpoints locaux et les dépôts Hugging Face, avec export JSON et Markdown.

## Script de benchmark d'inférence

```bash
python scripts/benchmark_infer.py --help
```

Le runtime par défaut reste court (warmup + 3 exécutions mesurées).

### Exemple local

```bash
python scripts/benchmark_infer.py \
  --source local \
  --checkpoint checkpoints/ckpt_last.pt \
  --meta data/processed/meta.json \
  --tokenizer char \
  --config configs/base.toml \
  --device cpu \
  --json-out outputs/bench_infer.json \
  --md-out outputs/bench_infer.md
```

### Exemple Hugging Face

```bash
python scripts/benchmark_infer.py \
  --source hf \
  --repo-id LabCoreAI/<id> \
  --config configs/base.toml \
  --device cuda \
  --json-out outputs/bench_infer_hf.json \
  --md-out outputs/bench_infer_hf.md
```

## Mesures effectuées

- Génération de warmup (`--warmup-tokens`, non comptée dans le débit final).
- Génération mesurée (`--gen-tokens`) répétée `--iters` fois.
- Résumé du débit: `mean`, `min`, `max` tokens/sec.
- Pic de VRAM (`torch.cuda.max_memory_allocated`) en exécution CUDA.

Les réglages de reproductibilité sont lus dans `[generation]` quand `--config` est fourni:

- `seed` (graine d'initialisation)
- `deterministic`
- réglages de sampling (`temperature`, `top_k`, `top_p`, `repetition_penalty`)
- `use_kv_cache` (sauf surcharge via flags CLI)

## Schéma de sortie JSON (résumé)

```json
{
  "timestamp": "...",
  "commit": "...",
  "platform": {"os": "...", "python": "..."},
  "torch": {"version": "...", "cuda": "..."},
  "device": {"type": "cpu|cuda", "name": "..."},
  "model": {"source": "local|hf", "params_m": 0.0, "block_size": 0, "n_layer": 0, "n_head": 0, "n_embd": 0},
  "generation": {"prompt": "...", "gen_tokens": 256, "temperature": 0.9, "top_k": 40, "top_p": 1.0, "repetition_penalty": 1.0, "use_kv_cache": true},
  "results": {"iters": 3, "tokens_per_sec": {"mean": 0.0, "min": 0.0, "max": 0.0}, "vram_peak_mib": null}
}
```

## Résultats communauté

Collez la ligne Markdown générée (via `--md-out` ou la sortie terminal) dans ce tableau.
Ajoutez le JSON dans la description de PR si disponible.

| Périphérique | Source | Taille du modèle (params M) | KV-cache | gen_tokens | mean tok/s | pic VRAM MiB |
|---|---|---:|---|---:|---:|---:|
| _votre résultat_ | _local/hf_ | _0.000_ | _on/off_ | _256_ | _0.00_ | _N/A ou valeur_ |
