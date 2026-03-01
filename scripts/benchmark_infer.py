# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from labcore_llm import GPT, GPTConfig, load_config
from labcore_llm.tokenizer import BPETokenizer, CharTokenizer
from labcore_llm.utils import configure_generation_reproducibility


def load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        fallback = Path("data/meta.json")
        if fallback.exists():
            return json.loads(fallback.read_text(encoding="utf-8"))
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_tokenizer(tokenizer_cfg: dict):
    tok_type = tokenizer_cfg.get("type", "char")
    if tok_type == "bpe":
        return BPETokenizer(encoding_name=tokenizer_cfg.get("encoding_name", "gpt2"))
    if tok_type == "char":
        if "vocab" not in tokenizer_cfg:
            raise ValueError("Char tokenizer requires a saved vocab.")
        return CharTokenizer.from_dict(tokenizer_cfg)
    raise ValueError(f"Unsupported tokenizer type: {tok_type}")


def _build_tokenizer_cfg(tokenizer_name: str, base_cfg: dict) -> dict:
    if tokenizer_name == "bpe":
        return {"type": "bpe", "encoding_name": base_cfg.get("encoding_name", "gpt2")}
    if tokenizer_name == "char":
        if "vocab" not in base_cfg:
            raise ValueError("Char tokenizer requires a saved vocab.")
        return {"type": "char", "vocab": base_cfg["vocab"]}
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_name}")


def load_local_model(
    checkpoint_path: Path,
    meta_path: Path,
    tokenizer_name: str | None,
    device: str,
) -> tuple[GPT, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # nosec B614
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval()

    checkpoint_tok_cfg = checkpoint.get("tokenizer", {"type": "char"})
    meta = load_meta(meta_path)
    meta_tok_cfg = meta.get("tokenizer", checkpoint_tok_cfg)

    resolved_tokenizer = tokenizer_name or meta_tok_cfg.get("type", checkpoint_tok_cfg.get("type", "char"))
    if resolved_tokenizer == "char":
        base_tok_cfg = meta_tok_cfg if "vocab" in meta_tok_cfg else checkpoint_tok_cfg
    else:
        base_tok_cfg = checkpoint_tok_cfg | meta_tok_cfg
    tokenizer = load_tokenizer(_build_tokenizer_cfg(resolved_tokenizer, base_tok_cfg))
    return model, tokenizer


def load_hf_model(
    repo_id: str,
    repo_revision: str,
    tokenizer_name: str | None,
    device: str,
) -> tuple[GPT, object]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to load remote models: pip install huggingface_hub") from exc
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Install safetensors to load remote models: pip install safetensors") from exc

    config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json", revision=repo_revision))
    tokenizer_path = Path(hf_hub_download(repo_id=repo_id, filename="tokenizer.json", revision=repo_revision))
    weights_path = Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=repo_revision))

    model_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = GPT(
        GPTConfig(
            vocab_size=model_cfg["vocab_size"],
            block_size=model_cfg["block_size"],
            n_layer=model_cfg["n_layer"],
            n_head=model_cfg["n_head"],
            n_embd=model_cfg["n_embd"],
            dropout=model_cfg.get("dropout", 0.1),
            bias=model_cfg.get("bias", True),
            use_rope=model_cfg.get("use_rope", False),
            use_flash=model_cfg.get("use_flash", False),
        )
    )
    model.load_state_dict(load_file(str(weights_path)), strict=False)
    model = model.to(device).eval()

    tokenizer_cfg = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    resolved_tokenizer = tokenizer_name or "bpe"
    tokenizer = load_tokenizer(_build_tokenizer_cfg(resolved_tokenizer, tokenizer_cfg))
    return model, tokenizer


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _get_git_commit(repo_root: Path) -> str | None:
    def _resolve_git_dir(root: Path) -> Path | None:
        dot_git = root / ".git"
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            content = dot_git.read_text(encoding="utf-8").strip()
            marker = "gitdir:"
            if content.lower().startswith(marker):
                git_dir = content[len(marker) :].strip()
                return (root / git_dir).resolve()
        return None

    def _read_ref(git_dir: Path, ref_name: str) -> str | None:
        ref_path = git_dir / ref_name
        if ref_path.exists():
            value = ref_path.read_text(encoding="utf-8").strip()
            if value:
                return value

        packed_refs = git_dir / "packed-refs"
        if not packed_refs.exists():
            return None
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("^"):
                continue
            parts = stripped.split(" ", maxsplit=1)
            if len(parts) == 2 and parts[1] == ref_name:
                return parts[0]
        return None

    commit_re = re.compile(r"^[0-9a-fA-F]{40}$")
    try:
        git_dir = _resolve_git_dir(repo_root)
        if git_dir is None:
            return None

        head_path = git_dir / "HEAD"
        if not head_path.exists():
            return None
        head = head_path.read_text(encoding="utf-8").strip()
        if not head:
            return None

        if head.startswith("ref:"):
            ref_name = head.split(":", maxsplit=1)[1].strip()
            commit = _read_ref(git_dir, ref_name)
        else:
            commit = head
    except Exception:
        return None

    if commit is None:
        return None
    commit = commit.strip()
    if not commit_re.fullmatch(commit):
        return None
    return commit


def _model_config_fields(model: GPT) -> dict:
    cfg = model.config
    return {
        "block_size": int(cfg.block_size),
        "n_layer": int(cfg.n_layer),
        "n_head": int(cfg.n_head),
        "n_embd": int(cfg.n_embd),
    }


def _count_params_m(model: torch.nn.Module) -> float:
    return float(sum(param.numel() for param in model.parameters()) / 1_000_000.0)


def _resolve_use_kv_cache(
    source: str,
    cli_value: bool | None,
    generation_cfg: dict,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if "use_kv_cache" in generation_cfg:
        return bool(generation_cfg["use_kv_cache"])
    return source == "local"


def _resolve_sampling_cfg(generation_cfg: dict) -> tuple[float, int, float, float]:
    temperature = float(generation_cfg.get("temperature", 0.9))
    top_k = int(generation_cfg.get("top_k", 40))
    top_p = float(generation_cfg.get("top_p", 1.0))
    repetition_penalty = float(generation_cfg.get("repetition_penalty", 1.0))
    return temperature, top_k, top_p, repetition_penalty


def benchmark_inference(
    model: GPT,
    tokenizer,
    *,
    device: torch.device,
    prompt: str,
    warmup_tokens: int,
    gen_tokens: int,
    iters: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    use_kv_cache: bool,
) -> dict:
    if gen_tokens <= 0:
        raise ValueError("gen_tokens must be > 0.")
    if iters <= 0:
        raise ValueError("iters must be > 0.")
    if warmup_tokens < 0:
        raise ValueError("warmup_tokens must be >= 0.")

    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt must produce at least one token.")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        if warmup_tokens > 0:
            _ = model.generate(
                x,
                max_new_tokens=warmup_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_kv_cache=use_kv_cache,
                stream=False,
            )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)

        per_iter_seconds: list[float] = []
        per_iter_tok_s: list[float] = []

        for _ in range(iters):
            _sync_cuda_if_needed(device)
            start = time.perf_counter()
            generated = model.generate(
                x,
                max_new_tokens=gen_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_kv_cache=use_kv_cache,
                stream=False,
            )
            _sync_cuda_if_needed(device)
            elapsed = time.perf_counter() - start

            if not isinstance(generated, torch.Tensor):
                raise TypeError("Benchmark expects tensor output from generate(stream=False).")
            generated_tokens = int(generated.size(1) - x.size(1))
            tok_s = generated_tokens / max(elapsed, 1e-12)
            per_iter_seconds.append(elapsed)
            per_iter_tok_s.append(tok_s)

    peak_mib = None
    if device.type == "cuda":
        peak_mib = float(torch.cuda.max_memory_allocated(device=device) / (1024.0**2))

    mean_tok_s = sum(per_iter_tok_s) / len(per_iter_tok_s)
    return {
        "iters": iters,
        "per_iter_seconds": per_iter_seconds,
        "per_iter_tokens_per_sec": per_iter_tok_s,
        "tokens_per_sec": {
            "mean": float(mean_tok_s),
            "min": float(min(per_iter_tok_s)),
            "max": float(max(per_iter_tok_s)),
        },
        "vram_peak_mib": peak_mib,
    }


def render_markdown_table(result_payload: dict) -> str:
    model_info = result_payload["model"]
    generation = result_payload["generation"]
    results = result_payload["results"]
    device = result_payload["device"]

    vram_peak = results["vram_peak_mib"]
    vram_cell = "N/A" if vram_peak is None else f"{vram_peak:.1f}"

    header = (
        "| Device | Source | Model size (params M) | KV-cache | gen_tokens | mean tok/s | peak VRAM MiB |\n"
        "|---|---|---:|---|---:|---:|---:|"
    )
    row = (
        f"| {device['name']} ({device['type']}) | {model_info['source']} | {model_info['params_m']:.3f} | "
        f"{'on' if generation['use_kv_cache'] else 'off'} | {generation['gen_tokens']} | "
        f"{results['tokens_per_sec']['mean']:.2f} | {vram_cell} |"
    )
    return f"{header}\n{row}\n"


def run_benchmark(args: argparse.Namespace) -> tuple[dict, str]:
    runtime_device = args.device
    if runtime_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        runtime_device = "cpu"
    device = torch.device(runtime_device)

    cfg = load_config(args.config) if args.config else {}
    generation_cfg = cfg.get("generation", {})

    configure_generation_reproducibility(
        seed=generation_cfg.get("seed"),
        deterministic=generation_cfg.get("deterministic", False),
    )

    tokenizer_name = args.tokenizer
    if args.source == "hf" and tokenizer_name is None:
        tokenizer_name = "bpe"

    if args.source == "local":
        model, tokenizer = load_local_model(
            checkpoint_path=Path(args.checkpoint),
            meta_path=Path(args.meta),
            tokenizer_name=tokenizer_name,
            device=str(device),
        )
    else:
        model, tokenizer = load_hf_model(
            repo_id=args.repo_id,
            repo_revision=args.repo_revision,
            tokenizer_name=tokenizer_name,
            device=str(device),
        )

    model.eval()

    temperature, top_k, top_p, repetition_penalty = _resolve_sampling_cfg(generation_cfg)
    use_kv_cache = _resolve_use_kv_cache(args.source, args.use_kv_cache, generation_cfg)
    if args.stream:
        print("Note: --stream requested, but benchmark forces stream=False for comparable timing.")

    benchmark = benchmark_inference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        warmup_tokens=args.warmup_tokens,
        gen_tokens=args.gen_tokens,
        iters=args.iters,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_kv_cache=use_kv_cache,
    )

    device_name = "CPU"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device=device)
    elif platform.processor():
        device_name = platform.processor()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _get_git_commit(Path(__file__).resolve().parents[1]),
        "platform": {
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "device": {
            "type": device.type,
            "name": device_name,
        },
        "model": {
            "source": args.source,
            "params_m": _count_params_m(model),
            **_model_config_fields(model),
        },
        "generation": {
            "prompt": args.prompt,
            "gen_tokens": args.gen_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "use_kv_cache": use_kv_cache,
            "stream": False,
        },
        "results": {
            "iters": benchmark["iters"],
            "tokens_per_sec": benchmark["tokens_per_sec"],
            "vram_peak_mib": benchmark["vram_peak_mib"],
        },
    }
    markdown = render_markdown_table(report)
    return report, markdown


def _write_outputs(
    report: dict,
    markdown: str,
    json_out: str | None,
    md_out: str | None,
) -> None:
    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON written to: {json_path}")
    if md_out:
        md_path = Path(md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        print(f"Markdown written to: {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible inference benchmark for LabCore LLM")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--checkpoint", default=None, help="Required for --source local.")
    parser.add_argument("--meta", default=None, help="Required for --source local.")
    parser.add_argument("--repo-id", default=None, help="Required for --source hf.")
    parser.add_argument("--repo-revision", default="main")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=256)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--use-kv-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.source == "local":
        if not args.checkpoint:
            parser.error("--checkpoint is required when --source local.")
        if not args.meta:
            parser.error("--meta is required when --source local.")
    if args.source == "hf" and not args.repo_id:
        parser.error("--repo-id is required when --source hf.")

    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens must be >= 0.")
    if args.gen_tokens <= 0:
        parser.error("--gen-tokens must be > 0.")
    if args.iters <= 0:
        parser.error("--iters must be > 0.")
    return args


def main() -> None:
    args = parse_args()
    report, markdown = run_benchmark(args)

    print("\nInference benchmark summary")
    print(markdown)

    _write_outputs(
        report=report,
        markdown=markdown,
        json_out=args.json_out,
        md_out=args.md_out,
    )


if __name__ == "__main__":
    main()
