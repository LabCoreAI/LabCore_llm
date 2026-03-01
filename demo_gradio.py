# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import torch

from labcore_llm import GPT, GPTConfig, load_config
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # nosec B614
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


def load_hf_model(repo_id: str, repo_revision: str, device: str):
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

    tokenizer = load_tokenizer(json.loads(tokenizer_path.read_text(encoding="utf-8")))
    return model, tokenizer


def load_generation_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        return load_config(path).get("generation", {})
    except Exception as exc:  # pragma: no cover - demo fallback path
        print(f"Warning: failed to load config defaults from {path}: {exc}")
        return {}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _build_chat_prompt(system_prompt: str, history: list[tuple[str, str]], user_prompt: str, max_history_turns: int) -> str:
    if max_history_turns > 0:
        turns = history[-max_history_turns:]
    else:
        turns = []

    parts: list[str] = []
    cleaned_system = system_prompt.strip()
    if cleaned_system:
        parts.append(f"<|system|>\n{cleaned_system}\n")
    for user_msg, assistant_msg in turns:
        parts.append(f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}\n")
    parts.append(f"<|user|>\n{user_prompt}\n<|assistant|>\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for LabCore LLM")
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_last.pt")
    parser.add_argument("--meta", default="data/processed/meta.json")
    parser.add_argument("--repo-id", default="GhostPunishR/labcore-llm-50M")
    parser.add_argument("--repo-revision", default="main")
    parser.add_argument("--config", default="configs/base.toml")
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
        model, tokenizer = load_hf_model(repo_id=args.repo_id, repo_revision=args.repo_revision, device=device)
    else:
        model, tokenizer = load_local_model(Path(args.checkpoint), Path(args.meta), device=device)

    generation_defaults = load_generation_defaults(args.config)
    default_temperature = float(_clamp(float(generation_defaults.get("temperature", 0.9)), 0.1, 2.0))
    default_top_k = int(_clamp(float(generation_defaults.get("top_k", 40)), 0.0, 200.0))
    default_top_p = float(_clamp(float(generation_defaults.get("top_p", 0.95)), 0.1, 1.0))
    default_rep_penalty = float(_clamp(float(generation_defaults.get("repetition_penalty", 1.0)), 1.0, 2.0))
    default_max_new_tokens = int(_clamp(float(generation_defaults.get("max_new_tokens", 160)), 8.0, 512.0))
    default_use_kv_cache = bool(generation_defaults.get("use_kv_cache", True))
    default_stream = bool(generation_defaults.get("stream", True))
    default_system_prompt = str(generation_defaults.get("system_prompt", ""))
    default_max_history_turns = int(_clamp(float(generation_defaults.get("max_history_turns", 6)), 0.0, 20.0))

    supports_kv_cache = args.source == "local" and hasattr(model, "forward_with_kv")
    source_note = "KV cache unsupported for current source." if not supports_kv_cache else "KV cache available."
    base_status = f"Source: {args.source} | Device: {device} | {source_note}"

    def _status_text(stream_enabled: bool, kv_requested: bool) -> str:
        kv_active = kv_requested and supports_kv_cache
        if kv_requested and not supports_kv_cache:
            kv_state = "off (unsupported)"
        else:
            kv_state = "on" if kv_active else "off"
        return f"{base_status} | Stream: {'on' if stream_enabled else 'off'} | KV cache: {kv_state}"

    def _chat_infer(
        user_prompt: str,
        history: list[tuple[str, str]] | None,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        use_kv_cache: bool,
        stream: bool,
        system_prompt: str,
        max_history_turns: int,
    ) -> Iterator[tuple[str, list[tuple[str, str]], list[tuple[str, str]], str]]:
        user_prompt = user_prompt.strip()
        history = list(history or [])
        max_history_turns = max(0, int(max_history_turns))
        trimmed_history = history[-max_history_turns:] if max_history_turns > 0 else []
        status_text = _status_text(stream_enabled=bool(stream), kv_requested=bool(use_kv_cache))

        if not user_prompt:
            yield "", history, history, status_text
            return

        full_prompt = _build_chat_prompt(
            system_prompt=system_prompt,
            history=trimmed_history,
            user_prompt=user_prompt,
            max_history_turns=max_history_turns,
        )
        prompt_ids = tokenizer.encode(full_prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        use_kv_cache_effective = bool(use_kv_cache and supports_kv_cache)
        assistant_text = ""
        if stream:
            stream_iter = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_kv_cache=use_kv_cache_effective,
                stream=True,
            )
            for token_id in stream_iter:
                assistant_text += tokenizer.decode([token_id])
                updated_history = trimmed_history + [(user_prompt, assistant_text)]
                yield "", updated_history, updated_history, status_text
            if not assistant_text:
                updated_history = trimmed_history + [(user_prompt, assistant_text)]
                yield "", updated_history, updated_history, status_text
            return

        output = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_kv_cache=use_kv_cache_effective,
            stream=False,
        )
        if not isinstance(output, torch.Tensor):  # pragma: no cover - defensive guard
            raise TypeError("Expected tensor output when stream=False.")
        new_token_ids = output[0].tolist()[len(prompt_ids) :]
        assistant_text = tokenizer.decode(new_token_ids)
        updated_history = trimmed_history + [(user_prompt, assistant_text)]
        yield "", updated_history, updated_history, status_text

    def _clear_chat() -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], str]:
        return "", [], [], base_status

    with gr.Blocks(title="LabCore LLM Demo") as demo:
        gr.Markdown("# LabCore LLM Demo")
        gr.Markdown(
            "Minimal chat with optional KV cache and streaming. "
            "History uses raw text markers: `<|system|>`, `<|user|>`, `<|assistant|>`."
        )

        chatbot = gr.Chatbot(label="Chat", height=420, type="tuples")
        prompt = gr.Textbox(label="Your message", lines=4, placeholder="Ask something...")
        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Accordion("Generation settings", open=True):
            with gr.Row():
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=default_temperature, step=0.05)
                top_k = gr.Slider(label="Top-k", minimum=0, maximum=200, value=default_top_k, step=1)
                top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=default_top_p, step=0.01)
            with gr.Row():
                repetition_penalty = gr.Slider(
                    label="Repetition penalty",
                    minimum=1.0,
                    maximum=2.0,
                    value=default_rep_penalty,
                    step=0.01,
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    minimum=8,
                    maximum=512,
                    value=default_max_new_tokens,
                    step=8,
                )
            with gr.Row():
                use_kv_cache = gr.Checkbox(label="Use KV cache", value=default_use_kv_cache)
                stream = gr.Checkbox(label="Stream output", value=default_stream)
            system_prompt = gr.Textbox(label="System prompt", lines=3, value=default_system_prompt)
            max_history_turns = gr.Slider(
                label="Max history turns",
                minimum=0,
                maximum=20,
                value=default_max_history_turns,
                step=1,
            )

        status = gr.Markdown(value=base_status)
        history_state = gr.State([])

        send_btn.click(
            fn=_chat_infer,
            inputs=[
                prompt,
                history_state,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                use_kv_cache,
                stream,
                system_prompt,
                max_history_turns,
            ],
            outputs=[prompt, chatbot, history_state, status],
        )
        prompt.submit(
            fn=_chat_infer,
            inputs=[
                prompt,
                history_state,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                use_kv_cache,
                stream,
                system_prompt,
                max_history_turns,
            ],
            outputs=[prompt, chatbot, history_state, status],
        )
        clear_btn.click(fn=_clear_chat, outputs=[prompt, chatbot, history_state, status])

    demo.queue().launch(server_port=args.port)


if __name__ == "__main__":
    main()
