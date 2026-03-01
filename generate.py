from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from labcore_llm import GPT, GPTConfig, load_config
from labcore_llm.tokenizer import BPETokenizer, CharTokenizer


def load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        fallback = Path("data/meta.json")
        if fallback.exists():
            return json.loads(fallback.read_text(encoding="utf-8"))
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_tokenizer_name(cli_name: str | None, config: dict | None, meta: dict) -> str:
    if cli_name:
        return cli_name
    if config:
        return config.get("general", {}).get("tokenizer", "char")
    return meta.get("tokenizer", {}).get("type", "char")


def load_tokenizer(meta: dict, tokenizer_name: str):
    tok_cfg = meta.get("tokenizer", {})
    if tokenizer_name == "bpe":
        return BPETokenizer(encoding_name=tok_cfg.get("encoding_name", "gpt2"))
    if tok_cfg.get("type") == "char" and "vocab" in tok_cfg:
        return CharTokenizer.from_dict(tok_cfg)
    raise ValueError("Char tokenizer requires vocab in meta.json. Run prepare_data.py with --tokenizer char.")


def configure_generation_reproducibility(seed: int | None, deterministic: bool) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Generation seed: {seed}")

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("Deterministic mode: enabled")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a LabCore checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_last.pt")
    parser.add_argument("--meta", default="data/processed/meta.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None)
    parser.add_argument("--prompt", default="To be")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    config = load_config(args.config) if args.config else None
    generation_cfg = config.get("generation", {}) if config else {}
    configure_generation_reproducibility(
        seed=generation_cfg.get("seed"),
        deterministic=generation_cfg.get("deterministic", False),
    )
    meta = load_meta(Path(args.meta))
    tokenizer_name = resolve_tokenizer_name(args.tokenizer, config, meta)
    tokenizer = load_tokenizer(meta, tokenizer_name)
    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=generation_cfg.get("top_p", 1.0),
            repetition_penalty=generation_cfg.get("repetition_penalty", 1.0),
        )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
