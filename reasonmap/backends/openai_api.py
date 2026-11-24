"""Backends that talk to an HTTP API.

* ``OpenAIChatBackend``: any OpenAI-compatible ``/v1/chat/completions`` server -- vLLM,
  SGLang, Volcengine Ark (Doubao), OpenAI chat models.
* ``OpenAIResponsesBackend``: OpenAI ``/v1/responses`` (o3, GPT-4o; also served by vLLM).
* ``GeminiBackend``: Google Gen AI SDK.
"""

import os
import time
from typing import Optional

from .base import Backend, encode_image, image_data_url

# Generation length used by the original open-source evaluation.
DEFAULT_MAX_TOKENS = 2048


def _retry(fn, retries, what):
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - surface the last error
            if attempt == retries:
                raise
            print(f"[retry {attempt + 1}/{retries}] {what}: {type(e).__name__}: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 60)


class OpenAIChatBackend(Backend):
    concurrent = True

    def __init__(self, model, family, base_url=None, api_key=None, max_tokens=DEFAULT_MAX_TOKENS,
                 sampling=None, max_image_pixels=None, image_detail=None, resample="bicubic",
                 timeout=3600, retries=3, extra_body=None):
        super().__init__(model, family)
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout, max_retries=0)
        self.max_tokens = max_tokens
        self.max_image_pixels = max_image_pixels
        self.image_detail = image_detail
        self.resample = resample
        self.retries = retries
        self.request_kwargs, self.extra_body = self._sampling_kwargs(family.gen_kwargs, sampling or {})
        if extra_body:
            self.extra_body.update(extra_body)

    @staticmethod
    def _sampling_kwargs(defaults, overrides):
        """Family defaults + CLI overrides -> (standard kwargs, extra_body).

        Unset values are not sent, so the server falls back to the model's generation config.
        """
        merged = {k: v for k, v in defaults.items() if k != "do_sample"}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        kwargs, extra = {}, {}
        for key, value in merged.items():
            if key in ("temperature", "top_p", "seed"):
                kwargs[key] = value
            elif key in ("top_k", "repetition_penalty", "min_p"):
                extra[key] = value
        return kwargs, extra

    def _messages(self, prompt, image_path):
        content = []
        if image_path:
            image_url = {"url": image_data_url(image_path, self.max_image_pixels, self.resample)}
            if self.image_detail:
                image_url["detail"] = self.image_detail
            content.append({"type": "image_url", "image_url": image_url})
        content.append({"type": "text", "text": prompt})
        messages = []
        if self.family.system_prompt:
            messages.append({"role": "system", "content": self.family.system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def generate(self, prompt: str, image_path: Optional[str]):
        kwargs = dict(model=self.model, messages=self._messages(prompt, image_path), **self.request_kwargs)
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = _retry(lambda: self.client.chat.completions.create(**kwargs), self.retries, self.model)
        message = response.choices[0].message
        # vLLM / SGLang reasoning parsers and Ark put the trace in a separate field
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        token_count = response.usage.completion_tokens if response.usage else 0
        return self.make_generation(message.content, token_count, reasoning=reasoning)


class OpenAIResponsesBackend(Backend):
    concurrent = True

    def __init__(self, model, family, base_url=None, api_key=None, max_tokens=None, sampling=None,
                 max_image_pixels=None, reasoning_effort=None, timeout=3600, retries=3):
        super().__init__(model, family)
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                             timeout=timeout, max_retries=0)
        self.max_tokens = max_tokens
        self.max_image_pixels = max_image_pixels
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self.sampling = {k: v for k, v in (sampling or {}).items() if k in ("temperature", "top_p") and v is not None}

    def generate(self, prompt: str, image_path: Optional[str]):
        content = [{"type": "input_text", "text": prompt}]
        if image_path:
            content.append({"type": "input_image", "image_url": image_data_url(image_path, self.max_image_pixels)})
        kwargs = dict(model=self.model, input=[{"role": "user", "content": content}], **self.sampling)
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
        if self.max_tokens:
            kwargs["max_output_tokens"] = self.max_tokens

        response = _retry(lambda: self.client.responses.create(**kwargs), self.retries, self.model)
        summaries = []
        for item in response.output or []:
            if getattr(item, "type", None) == "reasoning":
                for part in getattr(item, "summary", None) or []:
                    if getattr(part, "type", None) == "summary_text" and part.text:
                        summaries.append(part.text.strip())
        reasoning = "\n\n".join(summaries) or None
        token_count = response.usage.output_tokens if response.usage else 0
        return self.make_generation(response.output_text, token_count, reasoning=reasoning)


class GeminiBackend(Backend):
    concurrent = True

    def __init__(self, model, family, api_key=None, include_thoughts=True, max_image_pixels=None, retries=3):
        super().__init__(model, family)
        from google import genai

        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.include_thoughts = include_thoughts
        self.max_image_pixels = max_image_pixels
        self.retries = retries

    def generate(self, prompt: str, image_path: Optional[str]):
        import base64

        from google.genai import types

        contents = []
        if image_path:
            mime, b64 = encode_image(image_path, self.max_image_pixels)
            contents.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime))
        contents.append(prompt)
        config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(include_thoughts=self.include_thoughts))

        response = _retry(
            lambda: self.client.models.generate_content(model=self.model, config=config, contents=contents),
            self.retries, self.model,
        )
        parts = response.candidates[0].content.parts or []
        thoughts = [p.text for p in parts if getattr(p, "thought", False) and p.text]
        answer = "".join(p.text for p in parts if not getattr(p, "thought", False) and p.text)
        usage = response.usage_metadata
        token_count = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        return self.make_generation(answer, token_count, reasoning="\n\n".join(thoughts) or None)
