"""T0-1 package bootstrap entry.

Why this file now:
- We already have `llm.py` and `sampling_params.py` placeholders.
- Exporting both at package level aligns with the original public API shape.
- This keeps the one-file-per-change flow while making user imports cleaner.
"""

from .llm import LLM
from .sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]
