# nano-vllm-repro (Ticket T0-1)

This folder is an isolated scaffold for reproducing the very first step of nano-vllm.

## Scope of T0-1
- Initialize a standalone folder structure.
- Provide base project files (`README.md`, `LICENSE`, `requirements.txt`).
- Provide a minimal importable package layout.

## Quick check
Run from repository root:

```bash
PYTHONPATH=repro-first-ticket python -c "import nanovllm; print(nanovllm.__file__)"
```

Expected: printed path should point to `repro-first-ticket/nanovllm/__init__.py`.
