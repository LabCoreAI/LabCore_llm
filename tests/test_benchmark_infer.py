# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import json
from argparse import Namespace

import torch

from labcore_llm.model import GPT, GPTConfig
from scripts.benchmark_infer import run_benchmark


def _create_tiny_local_artifacts(tmp_path):
    cfg = GPTConfig(
        vocab_size=16,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    model = GPT(cfg).eval()

    vocab = sorted(set("ROMEO: \n"))
    checkpoint = {
        "model_config": model.config_dict,
        "model_state_dict": model.state_dict(),
        "tokenizer": {"type": "char", "vocab": vocab},
    }
    checkpoint_path = tmp_path / "ckpt_tiny.pt"
    torch.save(checkpoint, checkpoint_path)

    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        json.dumps({"tokenizer": {"type": "char", "vocab": vocab}}),
        encoding="utf-8",
    )
    return checkpoint_path, meta_path


def _benchmark_args(checkpoint_path, meta_path) -> Namespace:
    return Namespace(
        source="local",
        checkpoint=str(checkpoint_path),
        meta=str(meta_path),
        repo_id=None,
        repo_revision="main",
        tokenizer="char",
        config=None,
        prompt="ROMEO:",
        warmup_tokens=2,
        gen_tokens=4,
        iters=1,
        use_kv_cache=True,
        stream=False,
        json_out=None,
        md_out=None,
        device="cpu",
    )


def test_benchmark_runs_cpu(tmp_path):
    checkpoint_path, meta_path = _create_tiny_local_artifacts(tmp_path)
    report, markdown = run_benchmark(_benchmark_args(checkpoint_path, meta_path))

    assert report["device"]["type"] == "cpu"
    assert report["results"]["tokens_per_sec"]["mean"] > 0.0
    assert report["results"]["iters"] == 1
    assert "mean tok/s" in markdown


def test_json_schema_keys(tmp_path):
    checkpoint_path, meta_path = _create_tiny_local_artifacts(tmp_path)
    report, _ = run_benchmark(_benchmark_args(checkpoint_path, meta_path))

    assert "timestamp" in report
    assert "platform" in report
    assert "torch" in report
    assert "device" in report
    assert "model" in report
    assert "generation" in report
    assert "results" in report

    assert "tokens_per_sec" in report["results"]
    assert "mean" in report["results"]["tokens_per_sec"]
    assert "min" in report["results"]["tokens_per_sec"]
    assert "max" in report["results"]["tokens_per_sec"]
