# Dépannage

Utilisez cette page pour diagnostiquer rapidement les erreurs fréquentes de setup, de données et d'exécution.
Toutes les ancres ci-dessous sont référencées depuis les pages des guides.

## Installation

### Torch non installé {#torch-not-installed}

Symptômes: `ModuleNotFoundError: torch`, ou échec des scripts à l'import.

Correctif:

```bash
python -m pip install -e ".[torch,dev]"
```

Utilisez le sélecteur officiel si vous avez besoin d'un build CUDA spécifique:
<https://pytorch.org/get-started/locally/>

### CUDA non détecté {#cuda-not-detected}

Symptômes: `torch.cuda.is_available()` retourne `False`, ou fallback CPU dans les scripts.

Vérification rapide:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Points à vérifier:

- Driver NVIDIA installé et à jour
- Build PyTorch compatible avec votre runtime CUDA
- Même environnement Python utilisé pour l'installation et l'exécution

## Données et métadonnées

### Chemin des métadonnées incohérent {#meta-path-mismatch}

Symptômes: génération/entraînement incorrects ou impossibles à charger via les métadonnées tokenizer.

Mapping attendu:

- Pipeline `txt` -> `data/processed/meta.json`
- Pipeline `bin` -> `data/meta.json`

### Shards binaires introuvables {#binary-shards-not-found}

Symptômes: `Binary shards not found` avec `training.data_format = "bin"`.

Correctif:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format bin --output-dir data/processed
```

### Vocabulaire char manquant {#char-vocab-missing}

Symptômes: `Char tokenizer requires vocab in meta.json`.

Correctif:

```bash
python scripts/prepare_data.py --dataset tinyshakespeare --tokenizer char --output-format txt --output-dir data/processed
```

Puis passez le `--meta data/processed/meta.json` correspondant aux commandes de génération/export.

## Exécution

### Mémoire insuffisante {#oom-errors}

Réduire dans cet ordre:

1. `training.batch_size`
2. `model.block_size`
3. `training.gradient_accumulation_steps`
4. Taille du modèle / complexité du preset

### FlashAttention indisponible {#flashattention-not-available}

LabCore applique des fallbacks automatiques:

- FlashAttention (priorité, si disponible)
- Fallback SDPA PyTorch
- Fallback attention causale standard

### Problèmes de chemin et de policy Windows {#windows-path-policy}

- Activez le venv avant les commandes.
- Lancez les commandes depuis la racine du repo.
- Citez les chemins si des espaces sont présents.
- Si la policy PowerShell bloque les scripts, utilisez les commandes `python ...` directement.
