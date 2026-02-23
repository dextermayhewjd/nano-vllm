from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Sampling controls aligned with the first nano-vllm commit shape."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
