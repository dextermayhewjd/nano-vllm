"""Sampling parameters placeholder for T0-1 incremental bootstrap.

Why this file now:
- The original public API pairs `LLM` with `SamplingParams`.
- Defining this dataclass early locks in the minimal generation parameter surface.
- It enables future `__init__.py` export wiring and engine integration in separate one-file steps.
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Minimal sampling parameter scaffold."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
