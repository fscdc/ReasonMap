"""Inference backends and the factory used by ``infer.py``.

backend            what it runs
-----------------  ----------------------------------------------------------------
hf                 transformers, in-process (Kimi-VL / QVQ / Qwen2.5-VL / InternVL3 / Skywork-R1V)
vllm, sglang       an already running vLLM / SGLang server (OpenAI-compatible API)
openai             any OpenAI-compatible chat-completions endpoint (set --base-url)
openai-responses   OpenAI Responses API (o3, GPT-4o, ...)
gemini             Google Gen AI SDK

Closed models used in the paper are available as presets (``--preset Doubao-115`` etc.).
"""

import os

from ..models import get_family

DEFAULT_BASE_URLS = {
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
}

# Largest image any supported open model consumes (Qwen2.5-VL: 16384 * 28 * 28 pixels). Useful as a
# --max-image-pixels value to shrink requests; by default maps are sent untouched so the server's
# processor resizes them exactly like the transformers backend does.
DEFAULT_SERVER_MAX_IMAGE_PIXELS = 16384 * 28 * 28

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

_DOUBAO = dict(backend="openai", base_url=ARK_BASE_URL, api_key_env="ARK_API_KEY", max_tokens=None,
               max_image_pixels=36_000_000, image_detail="high", resample="bilinear")

# Closed models evaluated in the paper. Append "-onlytext" to any name for the text-only ablation.
PRESETS = {
    "Gemini": dict(backend="gemini", model="models/gemini-2.5-flash-preview-04-17", api_key_env="GEMINI_API_KEY"),
    "Doubao-115": dict(_DOUBAO, model="doubao-1-5-vision-pro-32k-250115"),
    "Doubao-415": dict(_DOUBAO, model="doubao-1-5-thinking-pro-m-250415"),
    "Doubao-428": dict(_DOUBAO, model="doubao-1-5-thinking-vision-pro-250428"),
    "o3": dict(backend="openai-responses", model="o3", reasoning_effort="medium", max_tokens=None,
               api_key_env="OPENAI_API_KEY"),
    "4o": dict(backend="openai-responses", model="gpt-4o", max_tokens=None, api_key_env="OPENAI_API_KEY"),
}

BACKENDS = ("hf", "vllm", "sglang", "openai", "openai-responses", "gemini")

ONLYTEXT_SUFFIX = "-onlytext"


def resolve_preset(name):
    base = name[: -len(ONLYTEXT_SUFFIX)] if name.endswith(ONLYTEXT_SUFFIX) else name
    if base not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; available: {sorted(PRESETS)} (optionally with {ONLYTEXT_SUFFIX})")
    return dict(PRESETS[base]), name.endswith(ONLYTEXT_SUFFIX)


def build_backend(cfg):
    """Create a backend from a resolved config dict (see ``infer.resolve_config``)."""
    backend = cfg["backend"]
    family = get_family(cfg["model"], cfg.get("model_family"))
    sampling = cfg.get("sampling") or {}
    api_key = cfg.get("api_key")
    if not api_key and cfg.get("api_key_env"):
        api_key = os.environ.get(cfg["api_key_env"])

    if backend == "hf":
        from .hf import HFBackend

        from .openai_api import DEFAULT_MAX_TOKENS

        return HFBackend(cfg["model"], family, cache_dir=cfg.get("cache_dir"),
                         max_new_tokens=cfg.get("max_tokens") or DEFAULT_MAX_TOKENS,
                         attn_implementation=cfg.get("attn_implementation"), sampling=sampling)

    if backend in ("vllm", "sglang", "openai"):
        from .openai_api import OpenAIChatBackend

        base_url = cfg.get("base_url") or DEFAULT_BASE_URLS.get(backend)
        if backend == "openai" and not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        return OpenAIChatBackend(cfg.get("served_model") or cfg["model"], family, base_url=base_url, api_key=api_key,
                                 max_tokens=cfg["max_tokens"], sampling=sampling,
                                 max_image_pixels=cfg.get("max_image_pixels"), image_detail=cfg.get("image_detail"),
                                 resample=cfg.get("resample", "bicubic"), timeout=cfg.get("timeout", 3600))

    if backend == "openai-responses":
        from .openai_api import OpenAIResponsesBackend

        return OpenAIResponsesBackend(cfg.get("served_model") or cfg["model"], family, base_url=cfg.get("base_url"),
                                      api_key=api_key, max_tokens=cfg["max_tokens"], sampling=sampling,
                                      max_image_pixels=cfg.get("max_image_pixels"),
                                      reasoning_effort=cfg.get("reasoning_effort"), timeout=cfg.get("timeout", 3600))

    if backend == "gemini":
        from .openai_api import GeminiBackend

        return GeminiBackend(cfg["model"], family, api_key=api_key, max_image_pixels=cfg.get("max_image_pixels"))

    raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")
