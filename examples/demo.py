"""Run recruitertriage against a handful of canned emails.

By default uses the zero-dep HeuristicLLM so this script works in CI
and on machines without transformers/torch installed.

Pass --smollm to use SmolLM2-360M-Instruct (requires
`pip install "recruitertriage[smollm]"`).

    python examples/demo.py
    python examples/demo.py --smollm
"""

from __future__ import annotations

import argparse
import json

from recruitertriage import HeuristicLLM, triage


SAMPLES = [
    (
        "Staff ML Engineer @ AcmeAI",
        "Hi Mukunda - we're hiring a Staff ML Engineer to lead agents. "
        "Comp band 250-320k base + equity. Remote PT/PST friendly. "
        "Are you open to a 15-min chat next week?",
    ),
    (
        "Java backend role",
        "Hello, we are sourcing for a Java backend developer at a "
        "Fortune 100 bank. Onsite Charlotte. Interested?",
    ),
    (
        "Lead gen partnership",
        "Hi, our agency can send you 50 verified candidate emails per week. "
        "Buy emails, save time. Reply YES for pricing.",
    ),
    (
        "Quick chat",
        "Hey - saw your GitHub. Could you send your resume if you're "
        "open to a chat? Let me know if interested!",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smollm", action="store_true",
                    help="Use SmolLM2-360M-Instruct (heavy import).")
    args = ap.parse_args()

    if args.smollm:
        from recruitertriage.smollm import make_smollm
        llm = make_smollm()
        backend = "SmolLM2-360M-Instruct"
    else:
        llm = HeuristicLLM()
        backend = "HeuristicLLM (zero-dep fallback)"

    print(f"backend: {backend}\n")

    for i, (subj, body) in enumerate(SAMPLES, 1):
        d = triage(subject=subj, body=body, llm=llm)
        print(f"[{i}] {subj}")
        print(f"    label={d.label.value:<11} conf={d.confidence:.2f}  reason={d.reason}")
        if d.suggested_reply:
            print(f"    reply: {d.suggested_reply}")
        print()


if __name__ == "__main__":
    main()
