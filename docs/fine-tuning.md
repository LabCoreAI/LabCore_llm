# Fine-Tuning

LabCore includes a LoRA instruction fine-tuning script for HF-compatible CausalLM models.

## Main Command

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

## Required Dependencies

```bash
pip install -e ".[finetune]"
```

Or for development + fine-tuning:

```bash
pip install -e ".[torch,dev,finetune]"
```

## Script Behavior

- Loads base model and tokenizer from `--model-id`.
- Auto-picks LoRA target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) when available.
- Normalizes examples to a stable instruction template.
- Tokenizes with fixed max sequence length.
- Trains with HF `Trainer`.
- Saves LoRA adapter and tokenizer to `--output-dir`.

## Dataset Field Mapping

The script supports common naming variants:

- instruction: `instruction` / `question` / `prompt`
- input: `input` / `context`
- output: `output` / `response` / `answer`

## Practical Tips

- Start with small `--max-samples` for validation.
- Tune `--batch-size` and `--max-seq-len` based on VRAM.
- Keep dataset quality high; noisy instruction pairs degrade output fast.

## Next Step

Review [Configuration Reference](configuration-reference.md) for full tuning controls.
