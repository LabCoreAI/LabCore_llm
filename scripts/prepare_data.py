# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from labcore_llm.tokenizer import BPETokenizer, CharTokenizer

TINY_SHAKESPEARE_FALLBACK_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _load_dataset_text(dataset_name: str) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Install HF dependencies to use datasets: pip install -e ".[hf]"') from exc

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    elif dataset_name == "tinyshakespeare":
        ds = load_dataset("tiny_shakespeare")
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    chunks: list[str] = []
    for split in ds.keys():
        split_ds = ds[split]
        if "text" not in split_ds.column_names:
            continue
        chunks.extend(split_ds["text"])
    merged = "\n\n".join(x for x in chunks if x)
    if not merged.strip():
        raise RuntimeError(f"No text rows found for dataset: {dataset_name}")
    return merged


def load_corpus(dataset_name: str, raw_dir: Path) -> str:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{dataset_name}.txt"
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8")

    if dataset_name == "tinyshakespeare":
        try:
            text = _load_dataset_text(dataset_name)
        except Exception:
            # Backward-compatible fallback if datasets mirror is unavailable.
            from urllib.request import urlopen

            with urlopen(TINY_SHAKESPEARE_FALLBACK_URL) as response:
                text = response.read().decode("utf-8")
    else:
        text = _load_dataset_text(dataset_name)

    raw_path.write_text(text, encoding="utf-8")
    return text


def split_text(text: str, val_ratio: float) -> tuple[str, str]:
    split = int((1.0 - val_ratio) * len(text))
    return text[:split], text[split:]


def _select_token_dtype(vocab_size: int) -> np.dtype:
    return np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32


def _tokenize(
    text: str,
    train_text: str,
    val_text: str,
    tokenizer_name: str,
) -> tuple[np.ndarray, np.ndarray, dict, int]:
    if tokenizer_name == "bpe":
        tokenizer = BPETokenizer()
        vocab_size = tokenizer.vocab_size
        token_dtype = _select_token_dtype(vocab_size)
        train_ids = np.array(tokenizer.encode(train_text), dtype=token_dtype)
        val_ids = np.array(tokenizer.encode(val_text), dtype=token_dtype)
        tokenizer_meta = {"type": "bpe", "encoding_name": tokenizer.encoding_name}
        return train_ids, val_ids, tokenizer_meta, vocab_size

    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    vocab_size = tokenizer.vocab_size
    token_dtype = _select_token_dtype(vocab_size)
    train_ids = np.array(tokenizer.encode(train_text), dtype=token_dtype)
    val_ids = np.array(tokenizer.encode(val_text), dtype=token_dtype)
    tokenizer_meta = tokenizer.to_dict()
    return train_ids, val_ids, tokenizer_meta, vocab_size


def _save_txt_dataset(
    output_dir: Path,
    text: str,
    train_text: str,
    val_text: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    meta: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (output_dir / "val.txt").write_text(val_text, encoding="utf-8")
    (output_dir / "corpus.txt").write_text(text, encoding="utf-8")
    np.save(output_dir / "train.npy", train_ids)
    np.save(output_dir / "val.npy", val_ids)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save_bin_dataset(
    output_dir: Path,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    meta: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(output_dir / "train.bin")
    val_ids.tofile(output_dir / "val.bin")
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_dataset(
    text: str,
    output_dir: Path,
    dataset_name: str,
    tokenizer_name: str,
    val_ratio: float,
    output_format: str,
) -> None:
    train_text, val_text = split_text(text, val_ratio)
    train_ids, val_ids, tokenizer_meta, vocab_size = _tokenize(
        text=text,
        train_text=train_text,
        val_text=val_text,
        tokenizer_name=tokenizer_name,
    )
    token_dtype = str(train_ids.dtype)
    meta = {
        "dataset": dataset_name,
        "vocab_size": vocab_size,
        "tokenizer": tokenizer_meta,
        "dtype": token_dtype,
        "output_format": output_format,
    }

    if output_format == "bin":
        bin_output_dir = output_dir.parent if output_dir.name == "processed" else output_dir
        meta.update(
            {
                "train_bin_file": "train.bin",
                "val_bin_file": "val.bin",
            }
        )
        _save_bin_dataset(
            output_dir=bin_output_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            meta=meta,
        )
        saved_dir = bin_output_dir
    else:
        meta.update(
            {
                "train_text_file": "train.txt",
                "val_text_file": "val.txt",
            }
        )
        _save_txt_dataset(
            output_dir=output_dir,
            text=text,
            train_text=train_text,
            val_text=val_text,
            train_ids=train_ids,
            val_ids=val_ids,
            meta=meta,
        )
        saved_dir = output_dir

    print(f"Saved dataset to {saved_dir.as_posix()}")
    print(f"train tokens: {len(train_ids)}")
    print(f"val tokens:   {len(val_ids)}")
    print(f"vocab size:   {vocab_size}")
    print(f"dtype:        {token_dtype}")
    print(f"tokenizer:    {tokenizer_name}")
    print(f"format:       {output_format}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare text data for LabCore LLM")
    parser.add_argument("--dataset", choices=["tinyshakespeare", "wikitext"], default="tinyshakespeare")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--output-format", choices=["txt", "bin"], default="txt")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    text = load_corpus(args.dataset, raw_dir)

    build_dataset(
        text,
        output_dir=output_dir,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        val_ratio=args.val_ratio,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
