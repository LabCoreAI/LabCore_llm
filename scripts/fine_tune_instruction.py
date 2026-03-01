# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import argparse
from pathlib import Path

from labcore_llm.config import load_config


def _pick_target_modules(model) -> list[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj"]
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    selected = [name for name in preferred if name in names]
    if selected:
        return selected
    fallback = [name for name in ["c_attn", "c_proj"] if name in names]
    return fallback or ["c_attn"]


def _example_to_text(example: dict) -> str:
    instruction = example.get("instruction") or example.get("question") or example.get("prompt") or ""
    input_text = example.get("input") or example.get("context") or ""
    output_text = example.get("output") or example.get("response") or example.get("answer") or ""
    if input_text:
        return f"### Instruction\n{instruction}\n\n### Input\n{input_text}\n\n### Response\n{output_text}"
    return f"### Instruction\n{instruction}\n\n### Response\n{output_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA instruction fine-tuning for LabCore-compatible CausalLM models.")
    parser.add_argument("--model-id", default="GhostPunishR/labcore-llm-50M")
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned", help="HF dataset path (Alpaca/OpenOrca-style).")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--model-revision", default="main", help="Model git ref (branch/tag/commit) on Hugging Face Hub.")
    parser.add_argument("--dataset-revision", default="main", help="Dataset git ref (branch/tag/commit) on Hugging Face Hub.")
    parser.add_argument("--output-dir", default="outputs/lora_instruction")
    parser.add_argument("--config", default="configs/bpe_rope_flash/bpe_50M_rope_flash.toml")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset-name", default=None, help="Optional dataset config name for load_dataset.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install fine-tuning dependencies first: pip install -e .[dev,finetune]"
        ) from exc

    cfg = load_config(Path(args.config))
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    opt_cfg = cfg.get("optimizer", {})

    max_seq_len = args.max_seq_len or model_cfg.get("block_size", 512)
    learning_rate = args.learning_rate or opt_cfg.get("learning_rate", train_cfg.get("learning_rate", 3e-4))
    batch_size = args.batch_size or train_cfg.get("batch_size", 4)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, revision=args.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.model_revision)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_pick_target_modules(model),
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset(args.dataset, args.dataset_name, split=args.dataset_split, revision=args.dataset_revision)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Final polish 2026: normalize Alpaca/OpenOrca-like rows into a stable instruction template.
    def preprocess(example: dict) -> dict:
        text = _example_to_text(example)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = ds.map(preprocess, remove_columns=ds.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=args.epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA fine-tuning completed. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
