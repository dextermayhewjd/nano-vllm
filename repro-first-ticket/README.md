# nano-vllm-repro (Ticket T0-1)

This folder is an isolated scaffold for reproducing the very first step of nano-vllm.

## Scope of T0-1
- Initialize a standalone folder structure.
- Provide base project files (`README.md`, `LICENSE`, `requirements.txt`).
- Provide a minimal importable package layout.

## Quick check
Run in `repro-first-ticket`:

```bash
python -c "import nanovllm; print(nanovllm.__file__)"
```

Expected: printed path should point to `repro-first-ticket/nanovllm/__init__.py`.


## Scope of T0-2
- Implement `Config` and `SamplingParams` dataclasses with first-commit-aligned defaults.
- Add minimal runtime assertions for early parameter validation.

## Quick check for T0-2
Run in `repro-first-ticket`:

```bash
python -c "from nanovllm import SamplingParams; from nanovllm.config import Config; print(SamplingParams())"
```


## Scope of T1-1
- Implement a **minimal** `SequenceStatus` and `Sequence` in `nanovllm.engine.sequence`.
- In this step only keep: token buffering, append behavior, and completion-token accounting (no full scheduler/block logic yet).

## Quick check for T1-1
Run in `repro-first-ticket`:

```bash
python - <<'PY'
from nanovllm.engine.sequence import Sequence

seq = Sequence([1, 2, 3])
seq.append_token(4)
print(seq.num_completion_tokens, seq.completion_token_ids)
PY
```


## T1-1 validation test
```bash
python -m unittest tests/test_sequence.py
```
