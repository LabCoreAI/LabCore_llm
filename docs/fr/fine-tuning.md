# Fine-Tuning

Le projet inclut un script LoRA pour fine-tuning instruction de modeles CausalLM compatibles HF.

## Commande principale

```bash
python scripts/fine_tune_instruction.py \
  --model-id GhostPunishR/labcore-llm-50M \
  --dataset yahma/alpaca-cleaned \
  --dataset-split train \
  --output-dir outputs/lora_instruction \
  --config configs/bpe_50M_rope_flash.toml \
  --max-samples 20000 \
  --epochs 1
```

## Dependances

```bash
pip install -e ".[finetune]"
```

Ou:

```bash
pip install -e ".[torch,dev,finetune]"
```

## Ce que fait le script

- charge model/tokenizer base
- detecte modules cibles LoRA
- normalise les exemples instruction
- tokenize avec taille sequence fixe
- entraine via HF `Trainer`
- sauvegarde adapter + tokenizer

## Mapping des champs dataset

- instruction: `instruction` / `question` / `prompt`
- input: `input` / `context`
- output: `output` / `response` / `answer`

## Conseils

- commencer avec peu de samples
- ajuster `batch-size` et `max-seq-len` selon VRAM
- privilegier un dataset propre et homogene

## Suite

Voir [Reference Config](configuration-reference.md).

