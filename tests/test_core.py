"""Tests for recruitertriage core triage logic.

These tests don't need transformers/torch — they exercise the
heuristic LM and the JSON parser directly."""

from __future__ import annotations

import json

import pytest

from recruitertriage import Decision, HeuristicLLM, Label, triage
from recruitertriage.core import _coerce_decision, _parse_llm_json, triage_batch


# ---- _parse_llm_json ------------------------------------------------------


def test_parse_clean_json():
    obj = _parse_llm_json('{"label":"interview","confidence":0.8}')
    assert obj == {"label": "interview", "confidence": 0.8}


def test_parse_json_buried_in_prose():
    raw = 'Sure! Here is the result:\n```json\n{"label":"reject"}\n```\nHope that helps!'
    obj = _parse_llm_json(raw)
    assert obj == {"label": "reject"}


def test_parse_garbage_returns_none():
    assert _parse_llm_json("totally not json") is None


# ---- _coerce_decision -----------------------------------------------------


def test_coerce_unknown_label_falls_back_to_unsure():
    d = _coerce_decision({"label": "yolo", "confidence": 0.9, "reason": "x"})
    assert d.label is Label.unsure


def test_coerce_clamps_confidence():
    d = _coerce_decision({"label": "interview", "confidence": 5.0, "reason": "x"})
    assert d.confidence == 1.0
    d2 = _coerce_decision({"label": "interview", "confidence": -1.0, "reason": "x"})
    assert d2.confidence == 0.0


def test_coerce_none_returns_unsure():
    d = _coerce_decision(None)
    assert d.label is Label.unsure
    assert d.confidence == 0.0


# ---- HeuristicLLM end-to-end ---------------------------------------------


@pytest.mark.parametrize(
    "subject,body,expected",
    [
        # spam
        (
            "We have leads",
            "We sell verified bitcoin investment opportunities to your team.",
            Label.spam,
        ),
        # reject (stack mismatch)
        (
            "Java backend role at BankCorp",
            "Hi, we're hiring a Java developer for our trading platform.",
            Label.reject,
        ),
        # interview (clear fit)
        (
            "Staff ML Engineer at AcmeAI",
            "Looking for a Staff ML Engineer to lead our agents team. "
            "Comp 250-320k base + equity.",
            Label.interview,
        ),
        # needs_info (vague but legit)
        (
            "Quick question",
            "Hey, would you be open to a chat? Send your resume "
            "if interested!",
            Label.needs_info,
        ),
    ],
)
def test_heuristic_llm_round_trip(subject: str, body: str, expected: Label):
    d = triage(subject=subject, body=body, llm=HeuristicLLM())
    assert d.label is expected
    assert 0.0 <= d.confidence <= 1.0
    assert d.reason


def test_unsure_when_no_signals():
    d = triage(
        subject="Hello",
        body="Hi, just saying hi.",
        llm=HeuristicLLM(),
    )
    assert d.label is Label.unsure


# ---- BYO-LLM contract -----------------------------------------------------


def test_byo_llm_callable_works():
    def fake_llm(prompt: str) -> str:
        return json.dumps({
            "label": "interview",
            "confidence": 0.91,
            "reason": "test fixture",
            "suggested_reply": "thanks, send details",
        })

    d = triage(subject="hi", body="hi", llm=fake_llm)
    assert d.label is Label.interview
    assert d.confidence == pytest.approx(0.91)
    assert d.suggested_reply == "thanks, send details"


def test_signals_captured():
    d = triage(subject="x", body="y", llm=HeuristicLLM())
    assert "prompt_chars" in d.signals
    assert "raw_chars" in d.signals
    assert isinstance(d.signals["prompt_chars"], int)


# ---- batch ---------------------------------------------------------------


def test_triage_batch():
    pairs = [
        ("Staff MLE role", "Senior Machine Learning Engineer at AcmeAI"),
        ("Java dev", "We need a Java backend dev"),
    ]
    results = triage_batch(pairs, llm=HeuristicLLM())
    assert len(results) == 2
    assert results[0].label is Label.interview
    assert results[1].label is Label.reject
