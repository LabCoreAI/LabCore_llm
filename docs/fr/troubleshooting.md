# Troubleshooting

Utilisez cette page pour diagnostiquer rapidement les erreurs frequentes de setup, donnees et runtime.
Toutes les ancres ci-dessous sont referencees depuis les pages guides.

## Setup

### Torch not installed {#torch-not-installed}

Symptomes: `ModuleNotFoundError: torch`, ou echec des scripts a l'import.

Correctif:

```bash
python -m pip install -e ".[torch,dev]"
```

Utilisez le selecteur officiel si vous avez besoin d'un build CUDA specifique:
<https://pytorch.org/get-started/locally/>

### CUDA not detected {#cuda-not-detected}

Symptomes: `torch.cuda.is_available()` retourne `False`, ou fallback CPU dans les scripts.

Verification rapide:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Points a verifier:

- Driver NVIDIA installe et a jour
- Build PyTorch compatible avec votre runtime CUDA
- Meme environnement Python utilise pour l'installation et l'execution

## Data and Metadata

### Meta path mismatch {#meta-path-mismatch}

Symptomes: generation/entrainement incorrects ou impossibles a charger via metadata tokenizer.

Mapping attendu:

- Pipeline `txt` -> `data/processed/meta.json`
- Pipeline `bin` -> `data/meta.json`

### Binary shards not found {#binary-shards-not-found}

Symptomes: `Binary shards not found` avec `training.data_format = "bin"`.

Correctif:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format bin --output-dir data/processed
```

### Char vocab missing {#char-vocab-missing}

Symptomes: `Char tokenizer requires vocab in meta.json`.

Correctif:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
```

Puis passez le `--meta data/processed/meta.json` correspondant aux commandes generation/export.

## Runtime

### Out of memory {#oom-errors}

Reduire dans cet ordre:

1. `training.batch_size`
2. `model.block_size`
3. `training.gradient_accumulation_steps`
4. Taille modele / complexite preset

### FlashAttention not available {#flashattention-not-available}

LabCore applique des fallbacks automatiques:

- FlashAttention (priorite, si disponible)
- Fallback SDPA PyTorch
- Fallback attention causale standard

### Windows path and policy issues {#windows-path-policy}

- Activez le venv avant les commandes.
- Lancez les commandes depuis la racine du repo.
- Citez les chemins si des espaces sont presents.
- Si la policy PowerShell bloque les scripts, utilisez les commandes `python ...` directement.
