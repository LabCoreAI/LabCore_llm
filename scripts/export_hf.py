from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from labcore_llm import GPT, GPTConfig


def load_meta(meta_path: Path) -> dict:
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    for fallback in (Path("data/processed/meta.json"), Path("data/meta.json")):
        if fallback.exists():
            return json.loads(fallback.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LabCore checkpoint to HF-friendly folder")
    parser.add_argument("--checkpoint", default="checkpoints/ckpt_last.pt")
    parser.add_argument("--meta", default="data/processed/meta.json")
    parser.add_argument("--output-dir", default="outputs/hf_export")
    parser.add_argument("--repo-id", default="GhostPunishR/labcore-llm-50M")
    parser.add_argument("--push", action="store_true", help="Push export folder to Hugging Face Hub")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = GPT(GPTConfig(**model_config))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    try:
        from safetensors.torch import save_model
    except ImportError as exc:
        raise RuntimeError("Install safetensors to export HF weights: pip install safetensors") from exc

    model_path = output_dir / "model.safetensors"
    # GPT ties lm_head and token embedding weights; save_model handles shared tensors.
    save_model(model, str(model_path))

    config_payload = {
        "architectures": ["GPT"],
        "model_type": "labcore_gpt",
        "vocab_size": model_config["vocab_size"],
        "block_size": model_config["block_size"],
        "n_layer": model_config["n_layer"],
        "n_head": model_config["n_head"],
        "n_embd": model_config["n_embd"],
        "dropout": model_config["dropout"],
        "bias": model_config["bias"],
        "use_rope": model_config.get("use_rope", False),
        "use_flash": model_config.get("use_flash", False),
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    meta_payload = load_meta(Path(args.meta))
    tokenizer_payload = meta_payload.get("tokenizer") or checkpoint.get("tokenizer", {"type": "char"})
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer_payload, indent=2), encoding="utf-8")

    card = f"""# LabCore LLM 50M

This folder is exported from LabCore LLM.

- Checkpoint: `{checkpoint_path.as_posix()}`
- RoPE: `{config_payload['use_rope']}`
- FlashAttention: `{config_payload['use_flash']}`
- Tokenizer: `{tokenizer_payload.get('type', 'unknown')}`
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")

    print(f"Exported files to {output_dir.as_posix()}")
    print(f"- {model_path.name}")
    print("- config.json")
    print("- tokenizer.json")
    print("- README.md")

    if args.push:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub to push exports: pip install huggingface_hub") from exc

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(output_dir), repo_id=args.repo_id, repo_type="model")
        print(f"Pushed export to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
