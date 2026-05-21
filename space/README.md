---
title: recruitertriage
emoji: "\U0001F4EC"
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
license: mit
tags:
  - small-models
  - smollm
  - hackathon
  - inbox
short_description: Triage recruiter outreach with a small (<1B) LM.
---

# recruitertriage

Triage recruiter outreach with a small (<1B) language model. Built for
the HuggingFace Build Small Hackathon.

The default backend is
[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).
It fits on a free Space CPU and still gets useful triage signal.

Each piece of recruiter outreach gets bucketed into one of five labels:

- `interview` — looks like a real fit, schedule a call
- `needs_info` — interesting but missing role/comp/level
- `reject` — clearly off (wrong stack, wrong level)
- `spam` — not a real role (vendor sales, lead-gen, scam)
- `unsure` — low confidence, falls through to a human

Source: https://github.com/MukundaKatta/recruitertriage
PyPI: https://pypi.org/project/recruitertriage/
