"""Core triage logic. No heavy deps — bring your own LM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable


# ---- decision shape -------------------------------------------------------


class Label(str, Enum):
    """Five buckets a recruiter outreach can fall into.

    `interview`  - looks like a strong fit, schedule a call
    `needs_info` - interesting but missing key details (role, comp, location)
    `reject`     - clearly off (wrong stack, mass blast, sketchy)
    `spam`       - not a real role / vendor pitch / sales
    `unsure`     - the model couldn't decide; fall through to human
    """

    interview = "interview"
    needs_info = "needs_info"
    reject = "reject"
    spam = "spam"
    unsure = "unsure"


@dataclass
class Decision:
    label: Label
    confidence: float  # 0..1
    reason: str
    suggested_reply: str | None = None
    signals: dict[str, object] = field(default_factory=dict)


LLM = Callable[[str], str]
"""A callable language model: takes a prompt string, returns raw text."""


# ---- prompt ---------------------------------------------------------------


_SYSTEM = (
    "You triage recruiter outreach. Pick exactly one label from: "
    "interview, needs_info, reject, spam, unsure. "
    "Return JSON ONLY with keys: label, confidence (0..1), reason (one short line), "
    "suggested_reply (1-3 sentences, plain, no AI phrases). "
    "If the email is vendor sales, pitch, or unrelated to a job role, return spam. "
    "If it lacks role title or comp range AND looks legit, return needs_info. "
    "If the stack/level is clearly wrong, return reject. "
    "If it looks like a real job match, return interview."
)


def _build_prompt(subject: str, body: str, hints: dict[str, object] | None) -> str:
    hints_block = ""
    if hints:
        hints_block = "\n\nUser hints:\n" + "\n".join(
            f"- {k}: {v}" for k, v in hints.items()
        )
    return (
        f"{_SYSTEM}\n\n"
        f"Subject: {subject}\n\n"
        f"Body:\n{body}{hints_block}\n\n"
        f"JSON:"
    )


# ---- parsing --------------------------------------------------------------


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_llm_json(raw: str) -> dict | None:
    """Pull the first JSON object out of a model's free-form text.

    Small models love to wrap things in prose, code fences, or extra
    commentary. Find the first {...} block and try to parse it."""
    # try whole-string first (well-behaved model)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _coerce_decision(obj: dict | None) -> Decision:
    if not isinstance(obj, dict):
        return Decision(Label.unsure, 0.0, "model output unparseable")

    label_raw = str(obj.get("label", "")).strip().lower()
    try:
        label = Label(label_raw)
    except ValueError:
        label = Label.unsure

    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    reason = str(obj.get("reason", "")).strip() or "no reason"
    reply = obj.get("suggested_reply")
    if reply is not None:
        reply = str(reply).strip() or None

    return Decision(label, conf, reason, suggested_reply=reply)


# ---- public entry ---------------------------------------------------------


def triage(
    *,
    subject: str,
    body: str,
    llm: LLM,
    hints: dict[str, object] | None = None,
) -> Decision:
    """Triage one piece of recruiter outreach.

    Pass any callable that maps prompt-string -> raw-string output.
    Small models (SmolLM2, Qwen-0.5B, etc.) are the target."""
    prompt = _build_prompt(subject=subject, body=body, hints=hints)
    raw = llm(prompt)
    parsed = _parse_llm_json(raw)
    decision = _coerce_decision(parsed)
    decision.signals["prompt_chars"] = len(prompt)
    decision.signals["raw_chars"] = len(raw)
    return decision


# ---- offline / heuristic fallback LM --------------------------------------


_INTERVIEW_HINTS = (
    "principal",
    "staff",
    "senior",
    "lead",
    "ml engineer",
    "machine learning",
    "ai engineer",
    "applied scientist",
    "research engineer",
)

_REJECT_HINTS = (
    "java",
    ".net",
    "salesforce admin",
    "qa manual",
    "support engineer",
    "tier 1 support",
)

_SPAM_HINTS = (
    "purchase order",
    "lead gen",
    "we have leads",
    "guaranteed candidates",
    "buy emails",
    "outsource your hiring",
    "verified bitcoin",
    "crypto investment",
)

_INFO_HINTS = (
    "more info",
    "let me know if interested",
    "send your resume",
    "open to a chat",
)


class HeuristicLLM:
    """Zero-dep fallback 'model'. Pattern-matches on keywords and emits
    a JSON-shaped string so the core pipeline still works without
    transformers/torch installed. Good for tests, CI, and offline demos.

    Don't ship this to prod. Plug in SmolLM2 or similar for real runs."""

    def __call__(self, prompt: str) -> str:
        body = prompt.lower()

        if any(s in body for s in _SPAM_HINTS):
            return json.dumps({
                "label": "spam",
                "confidence": 0.85,
                "reason": "matched spam keyword",
                "suggested_reply": None,
            })
        if any(s in body for s in _REJECT_HINTS):
            return json.dumps({
                "label": "reject",
                "confidence": 0.78,
                "reason": "stack/level mismatch",
                "suggested_reply": (
                    "Thanks for reaching out. Not the right fit for me right now."
                ),
            })
        if any(s in body for s in _INTERVIEW_HINTS):
            return json.dumps({
                "label": "interview",
                "confidence": 0.72,
                "reason": "title/level looks like a fit",
                "suggested_reply": (
                    "Thanks, this looks interesting. Could you share the role "
                    "level, comp band, and location?"
                ),
            })
        if any(s in body for s in _INFO_HINTS):
            return json.dumps({
                "label": "needs_info",
                "confidence": 0.6,
                "reason": "no role title or details given",
                "suggested_reply": (
                    "Happy to chat. Could you share the company, role title, "
                    "and comp band first?"
                ),
            })
        return json.dumps({
            "label": "unsure",
            "confidence": 0.3,
            "reason": "no clear signals",
        })


# ---- batch helpers --------------------------------------------------------


def triage_batch(
    emails: Iterable[tuple[str, str]],
    *,
    llm: LLM,
    hints: dict[str, object] | None = None,
) -> list[Decision]:
    """Triage a list of (subject, body) tuples. Pure convenience."""
    return [triage(subject=s, body=b, llm=llm, hints=hints) for s, b in emails]
