# Ajustement fin

Utilisez cette page pour du LoRA instruction tuning sur des checkpoints CausalLM compatibles HF.
Prérequis: dépendances HF + finetune et modèle de base accessible.

## Commandes

```bash
python scripts/fine_tune_instruction.py \
  --model-id GhostPunishR/labcore-llm-50M \
  --dataset yahma/alpaca-cleaned \
  --dataset-split train \
  --output-dir outputs/lora_instruction \
  --config configs/bpe_rope_flash/bpe_50M_rope_flash.toml \
  --max-samples 20000 \
  --epochs 1
```

Dépendances:

```bash
python -m pip install -e ".[torch,hf,finetune]"
```

## Fichiers de sortie / artefacts produits

- `outputs/lora_instruction/` (adapter LoRA + fichiers tokenizer)

## Mapping du dataset

Le script accepte plusieurs alias de champs:

- Instruction: `instruction`, `question`, `prompt`
- Input: `input`, `context`
- Output: `output`, `response`, `answer`

## Erreurs fréquentes

- Dépendances HF manquantes: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- OOM en fine-tuning: voir [Out of memory](troubleshooting.md#oom-errors).
- Incompatibilité modèle/tokenizer de base: vérifier la config et la compatibilité modèle avant lancement.

!!! note
    Le fine-tuning utilise la stack HF Trainer et écrit dans `outputs/` plutôt que dans `checkpoints/`.

## Suite / liens

- [Référence de configuration](configuration-reference.md)
- [Export et déploiement](export-and-deployment.md)
- [Opérations](operations.md)
