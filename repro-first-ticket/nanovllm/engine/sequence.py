"""Minimal sequence primitive for staged T1-1 delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Request lifecycle status used by scheduler."""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    """Minimal token buffer model.

    Design goal for this step: keep only fields needed for
    "create -> append -> count completion" workflow.
    """

    token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    seq_id: int = 0
    status: SequenceStatus = SequenceStatus.WAITING

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("token_ids must not be empty")

        self.token_ids = list(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)

        self.temperature = self.sampling_params.temperature
        self.max_tokens = self.sampling_params.max_tokens
        self.ignore_eos = self.sampling_params.ignore_eos

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]
