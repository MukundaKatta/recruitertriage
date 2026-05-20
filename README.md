# recruitertriage

[![PyPI](https://img.shields.io/pypi/v/recruitertriage.svg)](https://pypi.org/project/recruitertriage/)

![demo](docs/demo.gif)

Triage recruiter outreach with a small (<1B) language model. Built
for the [HuggingFace Build Small Hackathon][bs].

[bs]: https://huggingface.co/Build-Small-Hackathon

The default backend is
[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).
It fits on a free HF Space CPU and still gets useful triage signal.
You can swap in any callable LM (a fine-tune, Qwen-0.5B, a
heuristic, anything that maps `prompt -> string`).

## Install

```bash
pip install recruitertriage              # core only (zero heavy deps)
pip install "recruitertriage[smollm]"    # + SmolLM2 / transformers
pip install "recruitertriage[space]"     # + Gradio for the Space UI
```

Python 3.10+.

## What it does

Each piece of recruiter outreach gets bucketed into one of five labels:

| label        | meaning                                              |
|--------------|------------------------------------------------------|
| `interview`  | looks like a real fit, schedule a call               |
| `needs_info` | interesting but missing role/comp/level              |
| `reject`     | clearly off (wrong stack, wrong level)               |
| `spam`       | not a real role (vendor sales, lead-gen, scam)       |
| `unsure`     | low confidence — falls through to a human            |

Each `Decision` carries a label, a confidence (0..1), a one-line
reason, an optional suggested reply, and the raw signals the model
saw.

## Usage

```python
from recruitertriage import triage, HeuristicLLM

# Zero-dep fallback (good for tests/CI):
d = triage(
    subject="Staff ML Engineer @ AcmeAI",
    body="Hi - we're hiring a Staff ML Engineer. Comp 250-320k...",
    llm=HeuristicLLM(),
)
print(d.label, d.confidence, d.suggested_reply)

# Real small-model backend (requires the `smollm` extra):
from recruitertriage.smollm import make_smollm
llm = make_smollm("HuggingFaceTB/SmolLM2-360M-Instruct")
d = triage(subject="...", body="...", llm=llm)
```

`triage()` only cares that `llm` is `Callable[[str], str]`, so any
small instruction-tuned LM works. The core parser is tolerant of
small-model JSON quirks (code fences, prose around the object, etc.).

## HuggingFace Space

The `space/` directory is the deployable Gradio app:

```bash
pip install "recruitertriage[space]"
python space/app.py
```

To publish:

1. `hf login`
2. Create a new Gradio Space (CPU basic is enough)
3. Push the contents of `space/` as the Space root

## Demo

```bash
python examples/demo.py             # uses HeuristicLLM
python examples/demo.py --smollm    # uses SmolLM2-360M-Instruct
```

## Companion libraries

`recruitertriage` slots into the @mukundakatta agent-stack:

- [agentleash](https://github.com/MukundaKatta/agentleash) — USD/call budget cap + tool-arg gate
- [birddog](https://github.com/MukundaKatta/birddog) — audited Bright Data egress for scraping agents
- [agentvet](https://github.com/MukundaKatta/agentvet) — tool-arg validation with retry hints
- [agentsnap](https://github.com/MukundaKatta/agentsnap) — snapshot tests for agent traces

## License

MIT
