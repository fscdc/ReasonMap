"""transformers backend: runs the model in-process, one request at a time.

Supported families (same models as the original ``main.py`` / ``main_plus.py``):
Kimi-VL, QVQ, Qwen2.5-VL (incl. DeepEyes), InternVL3 and Skywork-R1V.
"""

import importlib.util
import logging
from typing import Optional

import torch
from PIL import Image

from .base import Backend

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

HF_FAMILIES = ("kimi-vl", "qvq", "qwen2.5-vl", "internvl", "skywork-r1v")


def _dtype_kwarg():
    """transformers >= 4.56 renamed ``torch_dtype`` to ``dtype``."""
    import transformers
    from packaging import version

    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    return {key: torch.bfloat16}


def _default_attn():
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


class HFBackend(Backend):
    concurrent = False

    def __init__(self, model, family, cache_dir=None, max_new_tokens=2048, attn_implementation=None,
                 sampling=None, device_map="auto"):
        super().__init__(model, family)
        if family.name not in HF_FAMILIES:
            raise ValueError(
                f"The transformers backend supports the families {HF_FAMILIES}; got {family.name!r} for {model}. "
                f"Pass --model-family, or serve the model with vLLM / SGLang and use --backend vllm/sglang."
            )
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = self._merge_sampling(family.gen_kwargs, sampling or {})
        self.attn = attn_implementation or _default_attn()
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.processor = None
        self.tokenizer = None
        self._load()

    # ------------------------------------------------------------------ loading

    @staticmethod
    def _merge_sampling(defaults, overrides):
        """Merge CLI sampling overrides into the family defaults (HF ``generate`` kwargs)."""
        kwargs = dict(defaults)
        for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if overrides.get(key) is not None:
                kwargs[key] = overrides[key]
        if overrides.get("temperature") is not None:
            kwargs["do_sample"] = overrides["temperature"] > 0
            if not kwargs["do_sample"]:
                for key in ("temperature", "top_p", "top_k"):
                    kwargs.pop(key, None)
        return kwargs

    def _from_pretrained(self, cls, **extra):
        common = dict(device_map=self.device_map, trust_remote_code=True, cache_dir=self.cache_dir, **_dtype_kwarg())
        attempts = [self.attn] + [a for a in ("sdpa", "eager") if a != self.attn]
        last_error = None
        for attn in attempts:
            try:
                model = cls.from_pretrained(self.model, attn_implementation=attn, **common, **extra)
                if attn != self.attn:
                    logger.warning("attn_implementation=%s unavailable for %s, using %s", self.attn, self.model, attn)
                self.attn = attn
                return model.eval()
            except (ValueError, ImportError) as e:  # unsupported attention implementation
                last_error = e
                if "attn" not in str(e).lower() and "attention" not in str(e).lower() and "flash" not in str(e).lower():
                    raise
        raise last_error

    def _load(self):
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        name = self.family.name
        if name == "kimi-vl":
            self.hf_model = self._from_pretrained(AutoModelForCausalLM)
            self.processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True, cache_dir=self.cache_dir)
        elif name == "qvq":
            self.hf_model = self._from_pretrained(transformers.Qwen2VLForConditionalGeneration)
            self.processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True, cache_dir=self.cache_dir)
        elif name == "qwen2.5-vl":
            self.hf_model = self._from_pretrained(transformers.Qwen2_5_VLForConditionalGeneration)
            self.processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True, cache_dir=self.cache_dir)
        else:  # internvl / skywork-r1v: remote-code models exposing .chat()
            use_flash = self.attn == "flash_attention_2"
            extra = dict(low_cpu_mem_usage=True, use_flash_attn=use_flash)
            if not use_flash:
                # their remote code only knows flash / eager attention
                self.attn = "eager"
            self.hf_model = self._from_pretrained(AutoModel, **extra)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True, cache_dir=self.cache_dir)

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else self.hf_model.device

    # ------------------------------------------------------------------ generation

    def generate(self, prompt: str, image_path: Optional[str]):
        name = self.family.name
        try:
            if name == "kimi-vl":
                text = self._generate_kimi(prompt, image_path)
            elif name in ("qvq", "qwen2.5-vl"):
                text = self._generate_qwen(prompt, image_path)
            else:
                text = self._generate_chat(prompt, image_path)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # token count = re-tokenised response, as in the original scripts
        tokenizer = self.tokenizer or self.processor.tokenizer
        token_count = len(tokenizer(text).input_ids)
        return self.make_generation(text, token_count)

    def _decode(self, inputs, generated_ids):
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _user_content(self, prompt, image_path):
        content = [{"type": "image", "image": image_path}] if image_path else []
        return content + [{"type": "text", "text": prompt}]

    def _generate_kimi(self, prompt, image_path):
        messages = [{"role": "user", "content": self._user_content(prompt, image_path)}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        images = [Image.open(image_path)] if image_path else None
        inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)
        generated = self.hf_model.generate(**inputs, max_new_tokens=self.max_new_tokens, **self.gen_kwargs)
        return self._decode(inputs, generated)

    def _generate_qwen(self, prompt, image_path):
        from qwen_vl_utils import process_vision_info

        messages = []
        if self.family.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.family.system_prompt}]})
        messages.append({"role": "user", "content": self._user_content(prompt, image_path)})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image_path:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs, video_inputs = None, None
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        generated = self.hf_model.generate(**inputs, max_new_tokens=self.max_new_tokens, **self.gen_kwargs)
        return self._decode(inputs, generated)

    def _generate_chat(self, prompt, image_path):
        from .internvl_image import load_image

        if image_path:
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            question = "<image>\n" + prompt
        else:
            # text-only: no pixel values and no <image> placeholder in the prompt
            pixel_values = None
            question = prompt
        generation_config = dict(max_new_tokens=self.max_new_tokens, **self.gen_kwargs)
        response = self.hf_model.chat(self.tokenizer, pixel_values, question, generation_config)
        del pixel_values
        return response
