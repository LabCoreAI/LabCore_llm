# Fine-Tuning

Utilisez cette page pour du LoRA instruction tuning sur checkpoints CausalLM compatibles HF.
Prerequis: dependances HF + finetune et modele de base accessible.

## Command(s)

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

Dependances:

```bash
python -m pip install -e ".[torch,hf,finetune]"
```

## Output Files / Artifacts Produced

- `outputs/lora_instruction/` (adapter LoRA + fichiers tokenizer)

## Dataset Mapping

Le script accepte plusieurs alias de champs:

- Instruction: `instruction`, `question`, `prompt`
- Input: `input`, `context`
- Output: `output`, `response`, `answer`

## Common Errors

- Dependances HF manquantes: voir [Torch not installed](troubleshooting.md#torch-not-installed).
- OOM en fine-tuning: voir [Out of memory](troubleshooting.md#oom-errors).
- Incompatibilite modele/tokenizer de base: verifier config et compatibilite modele avant lancement.

!!! note
    Le fine-tuning utilise la stack HF Trainer et ecrit dans `outputs/` plutot que `checkpoints/`.

## Next / Related

- [Configuration Reference](configuration-reference.md)
- [Export & Deployment](export-and-deployment.md)
- [Operations](operations.md)
