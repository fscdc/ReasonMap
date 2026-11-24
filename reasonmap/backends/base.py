"""Backend interface and shared helpers."""

import base64
import io
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

from PIL import Image

from ..models import ModelFamily, extract_final_answer

Image.MAX_IMAGE_PIXELS = None  # metro maps are very large


@dataclass
class Generation:
    response: str  # full output (reasoning + answer when the backend exposes both)
    final_answer: str  # text that gets scored
    token_count: int  # number of generated tokens
    reasoning: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class Backend(ABC):
    #: whether ``generate`` may be called from several threads at once
    concurrent = False

    def __init__(self, model: str, family: ModelFamily):
        self.model = model
        self.family = family

    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str]) -> Generation:
        """Answer ``prompt`` about the map at ``image_path`` (None for text-only runs)."""

    def make_generation(self, answer_text, token_count, reasoning=None):
        """Build a Generation from the answer text and an optional separate reasoning trace."""
        answer_text = answer_text or ""
        response = f"{reasoning}\n\n{answer_text}" if reasoning else answer_text
        return Generation(
            response=response,
            final_answer=extract_final_answer(answer_text, self.family),
            token_count=int(token_count or 0),
            reasoning=reasoning or None,
        )

    def close(self):
        pass


_RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}


@lru_cache(maxsize=16)
def encode_image(path: str, max_pixels: Optional[int] = None, resample: str = "bicubic"):
    """Return (mime_type, base64_str) for ``path``, downscaled to at most ``max_pixels``.

    The original bytes are sent untouched when no resize is needed. Results are cached
    because every sample of a city uses the same (large) map.
    """
    with Image.open(path) as img:
        width, height = img.size
        fmt = (img.format or "PNG").upper()
        if max_pixels and width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            resized = img.convert("RGB").resize(size, _RESAMPLE[resample])
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            return "image/png", base64.b64encode(buf.getvalue()).decode("ascii")
    with open(path, "rb") as f:
        data = f.read()
    mime = {"JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}.get(fmt, "image/png")
    return mime, base64.b64encode(data).decode("ascii")


def image_data_url(path: str, max_pixels: Optional[int] = None, resample: str = "bicubic") -> str:
    mime, b64 = encode_image(path, max_pixels, resample)
    return f"data:{mime};base64,{b64}"
