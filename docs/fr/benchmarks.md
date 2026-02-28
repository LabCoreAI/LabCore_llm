# Benchmarks

Utilisez cette page pour suivre des mesures de performance reproductibles selon preset et matériel.

## Modèle de Tableau

| Preset | Device | Batch / Seq / Accum | Débit (it/s ou tok/s) | VRAM Max | Notes |
|---|---|---|---:|---:|---|
| `configs/base.toml` | `cpu` | `32 / 128 / 1` | `0.00` | `0.0 GB` | placeholder |

## Exemple Minimal

Une ligne par run avec paramètres exacts :

| Preset | Device | Batch / Seq / Accum | Débit (it/s ou tok/s) | VRAM Max | Notes |
|---|---|---|---:|---:|---|
| `configs/bpe_rope_flash/bpe_30M_rope_flash.toml` | `cuda` | `8 / 512 / 8` | `TBD` | `TBD` | compléter après run stable |

## Conseils

- Faire quelques itérations de warmup avant mesure.
- Garder dataset, tokenizer et précision constants entre runs.
- Noter GPU, driver et version PyTorch dans les notes.
