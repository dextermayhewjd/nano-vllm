from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Placeholder sampling params for bootstrap stage."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
