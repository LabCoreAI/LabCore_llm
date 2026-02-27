from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_quant_binary(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "quantize",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = shutil.which("llama-quantize") or shutil.which("quantize")
    if found:
        return Path(found)
    raise FileNotFoundError("Could not find llama.cpp quantizer binary.")


def _run(command: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LabCore safetensors to GGUF and quantize.")
    parser.add_argument("--hf-dir", default="outputs/hf_export", help="Folder containing config.json + model.safetensors")
    parser.add_argument("--llama-cpp-dir", default="third_party/llama.cpp", help="Path to llama.cpp repository")
    parser.add_argument("--output-dir", default="outputs/gguf")
    parser.add_argument("--quant-type", choices=["Q4_K_M", "Q5_K_M", "all"], default="Q4_K_M")
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    llama_cpp_dir = Path(args.llama_cpp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"Missing converter script: {convert_script.as_posix()}. "
            "Clone/build llama.cpp first."
        )

    quant_bin = _resolve_quant_binary(llama_cpp_dir)
    f16_path = output_dir / "labcore-50m-f16.gguf"

    # Final polish 2026: keep an f16 GGUF source then emit compact quantized variants.
    _run(
        [
            sys.executable,
            str(convert_script),
            str(hf_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )

    quant_types = ["Q4_K_M", "Q5_K_M"] if args.quant_type == "all" else [args.quant_type]
    for quant_type in quant_types:
        out_file = output_dir / f"labcore-50m-{quant_type.lower()}.gguf"
        _run([str(quant_bin), str(f16_path), str(out_file), quant_type])
        print(f"Saved {out_file.as_posix()}")


if __name__ == "__main__":
    main()
