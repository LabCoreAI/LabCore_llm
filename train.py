from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from labcore_llm import GPT, GPTConfig, Trainer, TrainerConfig, load_config
from labcore_llm.tokenizer import BPETokenizer, CharTokenizer


def load_tokenizer(tokenizer_name: str, meta: dict, data_dir: Path):
    tok_cfg = meta.get("tokenizer", {})
    if tokenizer_name == "bpe":
        encoding_name = tok_cfg.get("encoding_name", "gpt2")
        return BPETokenizer(encoding_name=encoding_name)

    tokenizer = CharTokenizer()
    if tok_cfg.get("type") == "char" and "vocab" in tok_cfg:
        tokenizer = CharTokenizer.from_dict(tok_cfg)
    else:
        train_txt = data_dir / "train.txt"
        val_txt = data_dir / "val.txt"
        if train_txt.exists() and val_txt.exists():
            tokenizer.fit(train_txt.read_text(encoding="utf-8") + val_txt.read_text(encoding="utf-8"))
    return tokenizer


def load_data_sources(data_dir: Path, data_format: str):
    if data_format == "bin":
        train_bin = data_dir / "train.bin"
        val_bin = data_dir / "val.bin"
        if not train_bin.exists() or not val_bin.exists():
            raise FileNotFoundError(f"Binary shards not found in {data_dir.as_posix()}")
        return train_bin, val_bin

    train_txt = data_dir / "train.txt"
    val_txt = data_dir / "val.txt"
    if train_txt.exists() and val_txt.exists():
        return train_txt.read_text(encoding="utf-8"), val_txt.read_text(encoding="utf-8")
    return np.load(data_dir / "train.npy"), np.load(data_dir / "val.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LabCore GPT")
    parser.add_argument("--config", default="configs/base.toml")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    general_cfg = cfg.get("general", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    optimizer_cfg = cfg.get("optimizer", {})
    data_format = train_cfg.get("data_format", "txt")

    data_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    if data_format == "bin" and not (data_dir / "train.bin").exists() and (data_dir.parent / "train.bin").exists():
        data_dir = data_dir.parent
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tokenizer_name = args.tokenizer or general_cfg.get("tokenizer", "char")
    tokenizer = load_tokenizer(tokenizer_name, meta, data_dir)
    train_data, val_data = load_data_sources(data_dir, data_format=data_format)

    requested_device = args.device or train_cfg.get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        requested_device = "cpu"

    vocab_size = model_cfg.get("vocab_size")
    if vocab_size is None:
        vocab_size = meta.get("vocab_size")
    if vocab_size is None:
        vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("Unable to infer vocab_size from config, meta, or tokenizer.")

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg.get("block_size", 128),
        n_layer=model_cfg.get("n_layer", 6),
        n_head=model_cfg.get("n_head", 8),
        n_embd=model_cfg.get("n_embd", 256),
        dropout=model_cfg.get("dropout", 0.1),
        bias=model_cfg.get("bias", True),
        use_rope=model_cfg.get("use_rope", False),
        use_flash=model_cfg.get("use_flash", False),
    )

    model = GPT(gpt_config).to(requested_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg.get("learning_rate", train_cfg.get("learning_rate", 3e-4)),
        weight_decay=optimizer_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.01)),
        betas=(
            optimizer_cfg.get("beta1", 0.9),
            optimizer_cfg.get("beta2", 0.95),
        ),
    )

    max_iters = args.max_iters if args.max_iters is not None else train_cfg.get("max_iters", 2000)
    trainer_config = TrainerConfig(
        batch_size=train_cfg.get("batch_size", 32),
        block_size=gpt_config.block_size,
        max_iters=max_iters,
        eval_interval=train_cfg.get("eval_interval", 200),
        eval_iters=train_cfg.get("eval_iters", 50),
        log_interval=train_cfg.get("log_interval", 20),
        learning_rate=optimizer_cfg.get("learning_rate", train_cfg.get("learning_rate", 3e-4)),
        weight_decay=optimizer_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.01)),
        grad_clip=optimizer_cfg.get("grad_clip", train_cfg.get("grad_clip", 1.0)),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        warmup_iters=train_cfg.get("warmup_iters", 0),
        lr_decay_iters=train_cfg.get("lr_decay_iters", max_iters),
        min_lr=train_cfg.get("min_lr", 3e-5),
        save_interval=train_cfg.get("save_interval", train_cfg.get("eval_interval", 200)),
        beta1=optimizer_cfg.get("beta1", 0.9),
        beta2=optimizer_cfg.get("beta2", 0.95),
        tokenizer_name=tokenizer_name,
        data_format=data_format,
        meta_path=meta_path.as_posix(),
        device=requested_device,
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        config=trainer_config,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
