# HuggingFace Build Small — submission package (recruitertriage)

Pre-filled fields for the **HuggingFace Build Small Hackathon** (small models
<=32B). Deadline 2026-06-15; register by 2026-06-03 by joining the org via the
registration Space.

Event: https://huggingface.co/Build-Small-Hackathon
Register: https://huggingface.co/spaces/build-small-hackathon/registration

## Basic information

**Project Title**

    recruitertriage

**Tagline** (one line)

    Triage recruiter outreach into five buckets with a sub-1B language model
    that fits on a free CPU Space.

**Short description**

    recruitertriage labels recruiter messages as interview, needs_info,
    reject, spam, or unsure, with a confidence score, a one-line reason, and a
    suggested reply. The default backend is SmolLM2-360M-Instruct, small enough
    to run on a free HF Space CPU. A zero-dependency heuristic backend keeps it
    working with no model at all.

**Long description**

    Recruiter inboxes are mostly noise with a few real roles buried in it.
    recruitertriage runs each message through a tiny instruction-tuned model
    and returns a structured Decision:

      - label:        interview | needs_info | reject | spam | unsure
      - confidence:   0.0 to 1.0
      - reason:       one line on why this label
      - suggested_reply: an optional drafted response
      - signals:      the raw cues the model keyed on

    The point of the project is that you do NOT need a frontier model for this.
    SmolLM2-360M-Instruct (360M params) fits on a free HF Space CPU and still
    gets useful triage signal. The triage() function only requires the backend
    to be Callable[[str], str], so any small instruction-tuned LM (Qwen-0.5B, a
    fine-tune) or even a pure-Python heuristic drops in. The parser is tolerant
    of small-model JSON quirks (code fences, prose around the object), which is
    what makes a 360M model usable here.

    The 'unsure' label is a real escape hatch: low-confidence messages fall
    through to a human instead of being force-classified, so the small model
    never has to pretend it is certain.

**Why it fits "Build Small"**

    The whole design target is the smallest model that still does the job:
    360M params, free CPU Space, no GPU, and a zero-dep fallback so it degrades
    to a heuristic rather than failing. Small model, real task, runs anywhere.

**Technology & category tags**

    python, small-language-model, smollm2, smollm2-360m, huggingface,
    transformers, gradio, huggingface-spaces, text-classification,
    recruiter-triage, on-cpu, mit

## App hosting & code

**HuggingFace Space (live demo)**

    https://huggingface.co/spaces/mukunda1729/recruiter-triage
    (Gradio app, CPU basic, verified RUNNING)

**Public GitHub repository**

    https://github.com/MukundaKatta/recruitertriage

**PyPI package**

    https://pypi.org/project/recruitertriage/

## Submission requirement checklist

  - [x] Small model (<=32B): SmolLM2-360M-Instruct, 360M params
  - [x] Public Space link (above)
  - [x] Public code repo (above)
  - [ ] Demo video — USER TODO (record short walkthrough of the Space)
  - [ ] Social post — USER TODO (HF/X post linking the Space, per event rules)
  - [ ] Org registration — USER TODO (join via registration Space by 2026-06-03)

## Notes / open items

  - HF Space lives in the PERSONAL namespace (mukunda1729), not the org. Confirm
    the rules do not require org-namespace hosting before final submit.
  - HF handle: mukunda1729. CLI is `hf`.
