"""Model-family conventions shared by every backend.

A *family* captures what used to be hard-coded per model in ``main.py``: the system prompt,
default sampling settings, and how to cut the final answer out of the raw generation.
The same family applies whether the model runs under transformers, vLLM or SGLang.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelFamily:
    name: str
    system_prompt: Optional[str] = None
    # Everything up to (and including) the last of these markers is reasoning.
    think_end_markers: tuple = ()
    # QVQ puts its final answer in \boxed{\text{...}\text{...}}.
    boxed_text_answer: bool = False
    # Default sampling, HF-style keys (do_sample / temperature / top_p / top_k / repetition_penalty).
    gen_kwargs: dict = field(default_factory=dict)


QVQ_SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. "
    "You should think step-by-step."
)

FAMILIES = {
    "kimi-vl": ModelFamily("kimi-vl", think_end_markers=("◁/think▷",)),
    "qvq": ModelFamily("qvq", system_prompt=QVQ_SYSTEM_PROMPT, boxed_text_answer=True),
    "qwen2.5-vl": ModelFamily("qwen2.5-vl"),
    "internvl": ModelFamily("internvl", gen_kwargs={"do_sample": True}),
    "skywork-r1v": ModelFamily(
        "skywork-r1v",
        think_end_markers=("</think>",),
        gen_kwargs={"do_sample": True, "temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.05},
    ),
    # Anything else (e.g. a new model served by vLLM / SGLang, or a closed API).
    "generic": ModelFamily("generic", think_end_markers=("</think>", "◁/think▷")),
}


def detect_family(model: str) -> str:
    m = model.lower()
    if "kimi-vl" in m:
        return "kimi-vl"
    if "qvq" in m:
        return "qvq"
    if "skywork-r1v" in m:
        return "skywork-r1v"
    if "internvl" in m:
        return "internvl"
    if "qwen2.5-vl" in m or "qwen2_5_vl" in m or "deepeyes" in m:
        return "qwen2.5-vl"
    return "generic"


def get_family(model: str, override: Optional[str] = None) -> ModelFamily:
    name = override or detect_family(model)
    if name not in FAMILIES:
        raise ValueError(f"Unknown model family {name!r}; choose from {sorted(FAMILIES)}")
    return FAMILIES[name]


def _qvq_boxed_answer(text: str) -> str:
    match = re.search(r"boxed{(.+)}", text)
    if not match:
        return text
    boxed = match.group(1)
    parts = re.findall(r"\\text{(.+?)}", boxed)
    # Fall back to the raw boxed content when there is no \text{...} inside.
    return "\n".join(parts) if parts else boxed


def extract_final_answer(text: str, family: ModelFamily) -> str:
    """Strip the reasoning part of a generation and return the text that gets scored."""
    text = text or ""
    for marker in family.think_end_markers:
        if marker in text:
            text = text.split(marker)[-1]
    text = text.strip()
    if family.boxed_text_answer:
        text = _qvq_boxed_answer(text)
    return text
