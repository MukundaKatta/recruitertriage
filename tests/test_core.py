"""Tests for recruitertriage core triage logic.

These tests use only the Python standard library (``unittest``) and do
NOT require transformers/torch or pytest — they exercise the heuristic
LM, the JSON parser, and the public ``triage`` API directly.

Run them with::

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import json
import os
import sys
import unittest

# Make ``src/`` importable when running the tests in-place (i.e. without
# installing the package), so the suite works in plain CI checkouts.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from recruitertriage import Decision, HeuristicLLM, Label, triage  # noqa: E402
from recruitertriage.core import (  # noqa: E402
    _coerce_decision,
    _first_json_object,
    _parse_llm_json,
    triage_batch,
)


class ParseLlmJsonTests(unittest.TestCase):
    def test_parse_clean_json(self):
        obj = _parse_llm_json('{"label":"interview","confidence":0.8}')
        self.assertEqual(obj, {"label": "interview", "confidence": 0.8})

    def test_parse_json_buried_in_prose(self):
        raw = (
            "Sure! Here is the result:\n```json\n"
            '{"label":"reject"}\n```\nHope that helps!'
        )
        self.assertEqual(_parse_llm_json(raw), {"label": "reject"})

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(_parse_llm_json("totally not json"))

    def test_parse_empty_returns_none(self):
        self.assertIsNone(_parse_llm_json(""))

    def test_parse_nested_object_is_kept_whole(self):
        # Regression test: a non-greedy `{.*?}` regex would truncate this
        # at the first inner `}` and fail to parse. The balanced scanner
        # must keep the whole object intact.
        raw = 'Here you go: {"label": "interview", "meta": {"score": 1}}'
        self.assertEqual(
            _parse_llm_json(raw),
            {"label": "interview", "meta": {"score": 1}},
        )

    def test_parse_braces_inside_string_value(self):
        # Braces inside a quoted string must not confuse the depth count.
        raw = '{"reason": "use {curly} braces here", "label": "spam"}'
        self.assertEqual(
            _parse_llm_json(raw),
            {"reason": "use {curly} braces here", "label": "spam"},
        )

    def test_parse_ignores_trailing_prose(self):
        self.assertEqual(
            _parse_llm_json('{"label":"unsure"} hope this helps!'),
            {"label": "unsure"},
        )


class FirstJsonObjectTests(unittest.TestCase):
    def test_returns_none_when_no_brace(self):
        self.assertIsNone(_first_json_object("no object here"))

    def test_returns_first_balanced_object(self):
        self.assertEqual(
            _first_json_object('prefix {"a": {"b": 2}} suffix'),
            '{"a": {"b": 2}}',
        )

    def test_unbalanced_returns_none(self):
        self.assertIsNone(_first_json_object('{"a": 1'))


class CoerceDecisionTests(unittest.TestCase):
    def test_unknown_label_falls_back_to_unsure(self):
        d = _coerce_decision({"label": "yolo", "confidence": 0.9, "reason": "x"})
        self.assertIs(d.label, Label.unsure)

    def test_clamps_confidence_high(self):
        d = _coerce_decision({"label": "interview", "confidence": 5.0, "reason": "x"})
        self.assertEqual(d.confidence, 1.0)

    def test_clamps_confidence_low(self):
        d = _coerce_decision({"label": "interview", "confidence": -1.0, "reason": "x"})
        self.assertEqual(d.confidence, 0.0)

    def test_non_numeric_confidence_defaults_to_zero(self):
        d = _coerce_decision({"label": "spam", "confidence": "high", "reason": "x"})
        self.assertEqual(d.confidence, 0.0)

    def test_none_returns_unsure(self):
        d = _coerce_decision(None)
        self.assertIs(d.label, Label.unsure)
        self.assertEqual(d.confidence, 0.0)

    def test_missing_reason_gets_placeholder(self):
        d = _coerce_decision({"label": "interview", "confidence": 0.5})
        self.assertTrue(d.reason)

    def test_blank_suggested_reply_becomes_none(self):
        d = _coerce_decision(
            {"label": "interview", "confidence": 0.5, "suggested_reply": "  "}
        )
        self.assertIsNone(d.suggested_reply)

    def test_returns_decision_instance(self):
        d = _coerce_decision({"label": "spam", "confidence": 0.5})
        self.assertIsInstance(d, Decision)


class HeuristicRoundTripTests(unittest.TestCase):
    CASES = [
        (
            "We have leads",
            "We sell verified bitcoin investment opportunities to your team.",
            Label.spam,
        ),
        (
            "Java backend role at BankCorp",
            "Hi, we're hiring a Java developer for our trading platform.",
            Label.reject,
        ),
        (
            "Staff ML Engineer at AcmeAI",
            "Looking for a Staff ML Engineer to lead our agents team. "
            "Comp 250-320k base + equity.",
            Label.interview,
        ),
        (
            "Quick question",
            "Hey, would you be open to a chat? Send your resume if interested!",
            Label.needs_info,
        ),
    ]

    def test_round_trip_labels(self):
        for subject, body, expected in self.CASES:
            with self.subTest(subject=subject):
                d = triage(subject=subject, body=body, llm=HeuristicLLM())
                self.assertIs(d.label, expected)
                self.assertGreaterEqual(d.confidence, 0.0)
                self.assertLessEqual(d.confidence, 1.0)
                self.assertTrue(d.reason)

    def test_unsure_when_no_signals(self):
        d = triage(subject="Hello", body="Hi, just saying hi.", llm=HeuristicLLM())
        self.assertIs(d.label, Label.unsure)

    def test_heuristic_emits_parseable_json(self):
        # The heuristic backend must always emit valid JSON for the core
        # parser, regardless of which branch fires.
        llm = HeuristicLLM()
        for _, body, _ in self.CASES:
            with self.subTest(body=body):
                self.assertIsInstance(json.loads(llm(body.lower())), dict)


class TriageApiTests(unittest.TestCase):
    def test_byo_llm_callable_works(self):
        def fake_llm(prompt: str) -> str:
            return json.dumps(
                {
                    "label": "interview",
                    "confidence": 0.91,
                    "reason": "test fixture",
                    "suggested_reply": "thanks, send details",
                }
            )

        d = triage(subject="hi", body="hi", llm=fake_llm)
        self.assertIs(d.label, Label.interview)
        self.assertAlmostEqual(d.confidence, 0.91)
        self.assertEqual(d.suggested_reply, "thanks, send details")

    def test_unparseable_model_output_is_unsure(self):
        d = triage(subject="x", body="y", llm=lambda _p: "I cannot help with that")
        self.assertIs(d.label, Label.unsure)

    def test_empty_model_output_is_unsure(self):
        d = triage(subject="x", body="y", llm=lambda _p: "")
        self.assertIs(d.label, Label.unsure)

    def test_signals_captured(self):
        d = triage(subject="x", body="y", llm=HeuristicLLM())
        self.assertIn("prompt_chars", d.signals)
        self.assertIn("raw_chars", d.signals)
        self.assertIsInstance(d.signals["prompt_chars"], int)
        self.assertGreater(d.signals["prompt_chars"], 0)

    def test_hints_are_included_in_prompt(self):
        captured = {}

        def spy_llm(prompt: str) -> str:
            captured["prompt"] = prompt
            return '{"label": "unsure", "confidence": 0.0, "reason": "x"}'

        triage(subject="s", body="b", llm=spy_llm, hints={"my_stack": "python"})
        self.assertIn("my_stack", captured["prompt"])
        self.assertIn("python", captured["prompt"])


class TriageBatchTests(unittest.TestCase):
    def test_batch_preserves_order_and_labels(self):
        pairs = [
            ("Staff MLE role", "Senior Machine Learning Engineer at AcmeAI"),
            ("Java dev", "We need a Java backend dev"),
        ]
        results = triage_batch(pairs, llm=HeuristicLLM())
        self.assertEqual(len(results), 2)
        self.assertIs(results[0].label, Label.interview)
        self.assertIs(results[1].label, Label.reject)

    def test_empty_batch(self):
        self.assertEqual(triage_batch([], llm=HeuristicLLM()), [])


if __name__ == "__main__":
    unittest.main()
