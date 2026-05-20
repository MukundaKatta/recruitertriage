"""Gradio Space for recruitertriage on HuggingFace.

Built for the HuggingFace Build Small Hackathon. Uses
HuggingFaceTB/SmolLM2-360M-Instruct (under 1B params) as the underlying
language model. Fits on a free CPU Space.

To deploy:

    1. `huggingface-cli login` (or use `hf` CLI)
    2. Create a new Space (Gradio SDK, CPU basic)
    3. Push this `space/` directory as the Space root.
"""

from __future__ import annotations

import json
import os
import sys

# Make the local src/ importable when running the Space directly from
# this directory (the published Space pins recruitertriage as a normal
# pip dep via requirements.txt instead).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr

from recruitertriage import HeuristicLLM, triage


# Lazily build the SmolLM2 model so the Space starts fast and only
# pays the model-load cost on first request.
_smollm_cache = {"fn": None}


def _get_llm(backend: str):
    if backend == "heuristic":
        return HeuristicLLM()
    if _smollm_cache["fn"] is None:
        from recruitertriage.smollm import make_smollm
        _smollm_cache["fn"] = make_smollm()
    return _smollm_cache["fn"]


LABEL_EMOJI = {
    "interview": "yes",
    "needs_info": "ask",
    "reject": "no",
    "spam": "block",
    "unsure": "shrug",
}


def _do_triage(subject: str, body: str, backend: str):
    if not subject.strip() and not body.strip():
        return "Paste a recruiter email above.", "", ""

    llm = _get_llm(backend)
    d = triage(subject=subject, body=body, llm=llm)
    badge = f"{LABEL_EMOJI.get(d.label.value, '?')} {d.label.value} ({d.confidence:.0%})"
    detail = f"**Reason:** {d.reason}\n\n**Signals:** {json.dumps(d.signals)}"
    reply = d.suggested_reply or "_(no suggested reply)_"
    return badge, detail, reply


SAMPLES = [
    [
        "Staff ML Engineer @ AcmeAI",
        "Hi - we're hiring a Staff ML Engineer to lead agents. "
        "Comp band 250-320k base + equity. Remote PT/PST friendly. "
        "Are you open to a 15-min chat next week?",
    ],
    [
        "Quick question",
        "Hey, would you be open to a chat? Send your resume if "
        "interested!",
    ],
    [
        "Lead gen partnership",
        "Hi, our agency can send you 50 verified candidate emails per "
        "week. Buy emails, save time. Reply YES for pricing.",
    ],
]


with gr.Blocks(title="recruitertriage", theme="soft") as demo:
    gr.Markdown(
        "# recruitertriage\n"
        "Triage recruiter outreach with a small (<1B) language model. "
        "Built for the HuggingFace Build Small Hackathon. "
        "Backed by [SmolLM2-360M-Instruct]"
        "(https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)."
    )

    with gr.Row():
        with gr.Column():
            subject = gr.Textbox(label="Subject", lines=1)
            body = gr.Textbox(label="Body", lines=10)
            backend = gr.Radio(
                ["smollm", "heuristic"],
                value="smollm",
                label="Backend",
                info="smollm = SmolLM2-360M-Instruct.  heuristic = zero-dep keyword fallback (instant).",
            )
            run = gr.Button("Triage", variant="primary")
        with gr.Column():
            label = gr.Markdown(label="Decision")
            detail = gr.Markdown(label="Detail")
            reply = gr.Textbox(label="Suggested reply", lines=4)

    run.click(_do_triage, [subject, body, backend], [label, detail, reply])
    gr.Examples(examples=SAMPLES, inputs=[subject, body])


if __name__ == "__main__":
    demo.launch()
