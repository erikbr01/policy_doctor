"""Molmo2 (Ai2) vision-language backend — transformers ``AutoModelForImageTextToText``.

Works with checkpoints such as ``allenai/Molmo2-4B`` and ``allenai/Molmo2-8B``. Requires a
recent ``transformers`` (Molmo2 docs suggest ~4.57+) and ``trust_remote_code=True``."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from PIL import Image

from policy_doctor.vlm.backends.base import VLMBackend


class Molmo2Backend(VLMBackend):
    """Molmo2 multi-image understanding (see HF model card for message format)."""

    name = "molmo2"

    def __init__(
        self,
        model_id: str = "allenai/Molmo2-4B",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        # Reduce mode-collapse / repeated canned stories on captioning (optional; greedy if all null).
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        images_before_text: bool = True,
        **_kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype_name = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.images_before_text = images_before_text
        self._model = None
        self._processor = None

    def _extra_generate_kwargs(self) -> Dict[str, Any]:
        """HF generate() extras; only keys that are set."""
        out: Dict[str, Any] = {}
        if self.do_sample is not None:
            out["do_sample"] = bool(self.do_sample)
        if self.temperature is not None:
            out["temperature"] = float(self.temperature)
        if self.top_p is not None:
            out["top_p"] = float(self.top_p)
        if self.repetition_penalty is not None:
            out["repetition_penalty"] = float(self.repetition_penalty)
        return out

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "molmo / molmo2 backend requires `transformers` with "
                "`AutoModelForImageTextToText` (e.g. transformers>=4.57) and `torch`. "
                "Use backend=mock if unsupported."
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        td = dtype_map.get(self.torch_dtype_name, torch.bfloat16)

        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=td,
            device_map=None,
        )
        self._model.to(self.device)
        self._model.eval()

    def _model_device(self) -> Any:
        assert self._model is not None
        try:
            return self._model.device
        except Exception:
            return next(self._model.parameters()).device

    def describe_slice(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        import torch

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        text = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
        # Vision-first often reduces text-only completion of the instruction (mode collapse).
        content: list = []
        if self.images_before_text:
            for im in images:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": text})
        else:
            content.append({"type": "text", "text": text})
            for im in images:
                content.append({"type": "image", "image": im})

        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        dev = self._model_device()
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kw = {"max_new_tokens": self.max_new_tokens, **self._extra_generate_kwargs()}
        # Default when caller did not set sampling: light sampling helps caption diversity vs greedy.
        if "do_sample" not in gen_kw:
            gen_kw["do_sample"] = True
            gen_kw.setdefault("temperature", 0.4)
            gen_kw.setdefault("top_p", 0.9)
        # repetition_penalty is opt-in via backend_params: Molmo2+HF has triggered CUDA
        # device-side asserts when combined with default top-k warpers on some setups.

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kw)

        in_len = inputs["input_ids"].shape[1]
        gen_tokens = out_ids[0, in_len:]
        tok = getattr(self._processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("Molmo2 processor has no tokenizer for decode")
        return tok.decode(gen_tokens, skip_special_tokens=True).strip()

    def summarize_behavior_labels(
        self,
        *,
        cluster_id: int,
        slice_labels: Sequence[str],
        task_hint: str,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        import torch

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        text = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        dev = self._model_device()
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kw = {"max_new_tokens": self.max_new_tokens, **self._extra_generate_kwargs()}
        if "do_sample" not in gen_kw:
            gen_kw["do_sample"] = False

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kw)

        in_len = inputs["input_ids"].shape[1]
        gen_tokens = out_ids[0, in_len:]
        tok = getattr(self._processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("Molmo2 processor has no tokenizer for decode")
        return tok.decode(gen_tokens, skip_special_tokens=True).strip()

    def evaluate_slice_caption_coherency(
        self,
        *,
        cluster_id: int,  # noqa: ARG002
        slice_labels: Sequence[str],  # noqa: ARG002
        task_hint: str,  # noqa: ARG002
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        import torch

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        text = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        dev = self._model_device()
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kw = {"max_new_tokens": self.max_new_tokens, **self._extra_generate_kwargs()}
        if "do_sample" not in gen_kw:
            gen_kw["do_sample"] = False

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kw)

        in_len = inputs["input_ids"].shape[1]
        gen_tokens = out_ids[0, in_len:]
        tok = getattr(self._processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("Molmo2 processor has no tokenizer for decode")
        return tok.decode(gen_tokens, skip_special_tokens=True).strip()


def build_molmo2_backend(params: Optional[Dict[str, Any]] = None) -> Molmo2Backend:
    return Molmo2Backend(**(params or {}))


def build_molmo_backend(params: Optional[Dict[str, Any]] = None) -> Molmo2Backend:
    """Alias registry entry ``molmo`` → same class, ``name`` kept ``molmo`` for logs."""
    b = Molmo2Backend(**(params or {}))
    b.name = "molmo"
    return b
