from dataclasses import dataclass


@dataclass
class Config:
    """Configuration scaffold for bootstrap stage."""

    model: str = ""
