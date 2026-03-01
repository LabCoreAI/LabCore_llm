# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from labcore_llm import GPT, GPTConfig
from labcore_llm.tokenizer import BPETokenizer, CharTokenizer


def load_tokenizer(tokenizer_cfg: dict):
    tok_type = tokenizer_cfg.get("type", "char")
    if tok_type == "bpe":
        return BPETokenizer(encoding_name=tokenizer_cfg.get("encoding_name", "gpt2"))
    if tok_type == "char":
        if "vocab" not in tokenizer_cfg:
            raise ValueError("Char tokenizer requires a saved vocab.")
        return CharTokenizer.from_dict(tokenizer_cfg)
    raise ValueError(f"Unsupported tokenizer type: {tok_type}")


def load_local_model(checkpoint_path: Path, meta_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval()

    tokenizer_cfg = checkpoint.get("tokenizer", {"type": "char"})
    meta_candidates = [meta_path, Path("data/processed/meta.json"), Path("data/meta.json")]
    for candidate in meta_candidates:
        if candidate.exists():
            meta = json.loads(candidate.read_text(encoding="utf-8"))
            tokenizer_cfg = meta.get("tokenizer", tokenizer_cfg)
            break
    tokenizer = load_tokenizer(tokenizer_cfg)
    return model, tokenizer


def load_hf_model(repo_id: str, device: str):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to load remote models: pip install huggingface_hub") from exc
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Install safetensors to load remote models: pip install safetensors") from exc

    config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json"))
    tokenizer_path = Path(hf_hub_download(repo_id=repo_id, filename="tokenizer.json"))
    weights_path = Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors"))

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

    tokenizer = load_tokenizer(json.loads(tokenizer_path.read_text(encoding="utf-8")))
    return model, tokenizer


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove_mask = cumulative > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    sorted_logits[remove_mask] = float("-inf")
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: str,
) -> str:
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x[:, -model.config.block_size :])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")
            logits = top_p_filter(logits, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_id), dim=1)
    return tokenizer.decode(x[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for LabCore LLM")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_last.pt")
    parser.add_argument("--meta", default="data/processed/meta.json")
    parser.add_argument("--repo-id", default="GhostPunishR/labcore-llm-50M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to launch the demo: pip install gradio") from exc

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    if args.source == "hf":
        model, tokenizer = load_hf_model(repo_id=args.repo_id, device=device)
    else:
        model, tokenizer = load_local_model(Path(args.checkpoint), Path(args.meta), device=device)

    def _infer(prompt: str, temperature: float, top_k: int, top_p: float, max_new_tokens: int) -> str:
        return generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )

    with gr.Blocks(title="LabCore LLM Demo") as demo:
        gr.Markdown("# LabCore LLM - 50M RoPE/Flash Demo")
        prompt = gr.Textbox(label="Prompt", lines=5, value="In the beginning")
        with gr.Row():
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.9, step=0.05)
            top_k = gr.Slider(label="Top-k", minimum=0, maximum=200, value=40, step=1)
            top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.95, step=0.01)
        max_new_tokens = gr.Slider(label="Max new tokens", minimum=8, maximum=512, value=160, step=8)
        generate_btn = gr.Button("Generate")
        output = gr.Textbox(label="Generated text", lines=14)
        generate_btn.click(
            fn=_infer,
            inputs=[prompt, temperature, top_k, top_p, max_new_tokens],
            outputs=[output],
        )

    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
