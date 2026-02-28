# Fine-Tuning

Use this page for LoRA instruction tuning on HF-compatible CausalLM checkpoints.
Prerequisites: HF + finetune dependencies and an accessible base model.

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

Dependencies:

```bash
python -m pip install -e ".[torch,hf,finetune]"
```

## Output Files / Artifacts Produced

- `outputs/lora_instruction/` (LoRA adapter + tokenizer files)

## Dataset Mapping

The script accepts common field aliases:

- Instruction: `instruction`, `question`, `prompt`
- Input: `input`, `context`
- Output: `output`, `response`, `answer`

## Common Errors

- Missing HF dependencies: see [Torch not installed](troubleshooting.md#torch-not-installed).
- OOM during fine-tuning: see [Out of memory](troubleshooting.md#oom-errors).
- Wrong base model/tokenizer expectations: verify config and model compatibility before launching.

!!! note
    Fine-tuning uses the HF trainer stack and writes to `outputs/` rather than `checkpoints/`.

## Next / Related

- [Configuration Reference](configuration-reference.md)
- [Export & Deployment](export-and-deployment.md)
- [Operations](operations.md)
