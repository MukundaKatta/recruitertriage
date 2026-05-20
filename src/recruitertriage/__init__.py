"""recruitertriage - triage recruiter outreach with a small (<1B) LM.

A tiny, dependency-light core. Bring your own callable language model
(SmolLM2, Qwen-0.5B, your own fine-tune, or even a heuristic).

Example:

    from recruitertriage import triage, Decision

    def my_llm(prompt: str) -> str:
        # call SmolLM2-360M, return raw text
        ...

    result: Decision = triage(
        subject="Senior ML role at Acme",
        body="Hi! Are you open to a quick chat about a Senior MLE role...",
        llm=my_llm,
    )
    print(result.label, result.confidence, result.reason)

For the HuggingFace Build Small Hackathon entry, see `space/app.py`
for a Gradio UI bundled with SmolLM2-360M-Instruct.
"""

from __future__ import annotations

from .core import (
    Decision,
    HeuristicLLM,
    Label,
    triage,
)

__all__ = [
    "Decision",
    "HeuristicLLM",
    "Label",
    "triage",
]

__version__ = "0.1.0"
