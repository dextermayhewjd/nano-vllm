"""T0-1 package bootstrap entry.

Why this file now:
- T0-1 is about project bootstrap, and a Python package needs an explicit entry.
- This unlocks incremental follow-up files (llm.py, sampling_params.py) without changing more than one file per step.
- Keeps scope minimal and aligned with the one-file-per-change workflow.
"""

__all__: list[str] = []
