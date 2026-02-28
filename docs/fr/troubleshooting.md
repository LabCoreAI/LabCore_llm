# Dépannage

Problèmes fréquents de setup/runtime avec correctifs rapides.

## Installation PyTorch (CPU/CUDA)

Utilisez le sélecteur officiel selon OS + version CUDA :

<https://pytorch.org/get-started/locally/>

Commande type :

```bash
pip install -e ".[torch]"
```

## CUDA Non Détecté (`torch.cuda.is_available() == False`)

Vérifiez :

- Build PyTorch compatible avec votre runtime/driver CUDA.
- Driver NVIDIA installé et à jour.
- Même venv/Python utilisé pour installer et exécuter PyTorch.

Vérification rapide :

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## OOM: Quoi Réduire en Premier

En cas d'erreur mémoire, réduire dans cet ordre :

1. `training.batch_size`
2. `model.block_size` (longueur de séquence)
3. `training.gradient_accumulation_steps`
4. Stratégie de précision (si vous ajoutez mixed precision)

Si nécessaire, utiliser un preset de taille plus petite.

## Emplacement de `meta.json`

L'emplacement dépend du format de sortie :

- Pipeline `txt`: généralement `data/processed/meta.json`
- Pipeline `bin`: généralement `data/meta.json`

Si `train.py` ne trouve pas la metadata, vérifier `processed_dir` dans le TOML et le dossier de sortie réel de `prepare_data.py`.

## FlashAttention Non Disponible

LabCore applique un fallback automatiquement :

- Priorité: kernel FlashAttention (si dispo)
- Fallback: PyTorch SDPA
- Dernier chemin: attention causale standard

Vous pouvez garder `use_flash = true`, le fallback gère les environnements non supportés.

## Notes Windows

- Activer le venv avant les commandes.
- Citer les chemins si espaces.
- Exécuter depuis la racine du repo pour conserver les chemins relatifs.
- Si execution policy bloque les scripts, utiliser `python ...` directement.
