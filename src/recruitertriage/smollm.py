"""SmolLM2 (HuggingFaceTB/SmolLM2-360M-Instruct) integration.

Lives behind an extra: `pip install "recruitertriage[smollm]"`.

The default model is small enough to run on a laptop CPU; for the
HuggingFace Build Small Hackathon we explicitly target sub-1B
parameter models. Pass any other instruction-tuned causal LM by name."""

from __future__ import annotations

from typing import Any


def make_smollm(
    model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    *,
    device: str | None = None,
    max_new_tokens: int = 220,
    temperature: float = 0.2,
):
    """Build a callable LM around a HuggingFace causal model.

    Returns a function `(prompt: str) -> str` that you can pass straight
    into `recruitertriage.triage(..., llm=...)`."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(
            'recruitertriage[smollm] extras are required for SmolLM2. '
            'Install with: pip install "recruitertriage[smollm]"'
        ) from e

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    def _call(prompt: str) -> str:
        # SmolLM2-Instruct expects a chat template; the prompt becomes
        # a single user message. We deliberately keep the system message
        # baked into `prompt` so callers can swap models without breaking
        # the template.
        messages = [{"role": "user", "content": prompt}]
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs: dict[str, Any] = tok(rendered, return_tensors="pt").to(device)
        with torch_no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                pad_token_id=tok.eos_token_id,
            )
        # only decode the newly-generated tokens
        new = out[0, inputs["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip()

    return _call


def torch_no_grad():
    """Lazy torch.no_grad() so importing this module is cheap when torch
    isn't installed (the make_smollm call above already imported it)."""
    import torch  # noqa: PLC0415

    return torch.no_grad()
