# Support

## Where to Ask for Help

- Usage questions: open a GitHub Discussion (if enabled) or an issue with label `support`.
- Bug reports: use the bug issue template.
- Feature requests: use the feature request template.
- Security concerns: follow [SECURITY.md](SECURITY.md).

## Before Opening a Support Request

- Read [README.md](README.md) and configuration examples in `configs/`.
- Confirm dependencies are installed:

```bash
python -m pip install -e ".[torch,dev]"
```

- Re-run with a clean virtual environment if possible.
- Include exact command, logs, and environment info.

## Useful Environment Details

- OS and version
- Python version
- PyTorch version
- CUDA availability (`python -c "import torch; print(torch.cuda.is_available())"`)
- GPU model and VRAM (if training)
