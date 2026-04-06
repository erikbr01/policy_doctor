"""Qwen *-VL backends: Qwen2-VL, Qwen2.5-VL, Qwen3-VL (lazy transformers imports).

Registry names ``qwen2_vl`` and ``qwen3_vl`` both use this module; the HF class is
picked from ``model_id`` (e.g. ``Qwen/Qwen3-VL-4B-Instruct``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

from PIL import Image

from policy_doctor.vlm.backends.base import VLMBackend

# Generous HF image-processor caps (total pixels per image before shrink). ~16M px; lower if OOM.
_DEFAULT_IMAGE_MAX_PIXELS_HIGH_DETAIL = 16 * 16 * 4 * 16384
_DEFAULT_IMAGE_MIN_PIXELS_HIGH_DETAIL = 4 * 4 * 32 * 32  # match qwen_vl_utils floor spirit; avoid tiny upscales


def _normalize_qwen_frame_input(value: Any, *, legacy_use_video: Optional[bool]) -> str:
    """Return ``images`` or ``video``; *legacy_use_video* applies only when *value* is unset."""
    if value is not None and str(value).strip() != "":
        s = str(value).strip().lower()
        if s in ("images", "image", "multi_image", "multi-image", "bag"):
            return "images"
        if s in ("video", "frame_list", "frame-list"):
            return "video"
        raise ValueError(
            f"backend_params.qwen_frame_input must be 'images' or 'video', got {value!r}"
        )
    if legacy_use_video is None:
        return "images"
    return "video" if bool(legacy_use_video) else "images"


def _resolve_vl_model_class(model_id: str) -> Type[Any]:
    """Return the HF *ForConditionalGeneration class for this checkpoint id."""
    mid = model_id.lower()
    # NVIDIA Cosmos-Reason2-* is Qwen3-VL–family (see HF model card); do not route to Qwen2VL.
    if "cosmos-reason2" in mid or "cosmos_reason2" in mid:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if "qwen3-vl" in mid or "qwen3_vl" in mid:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if "2.5-vl" in mid or "2_5_vl" in mid:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    from transformers import Qwen2VLForConditionalGeneration

    return Qwen2VLForConditionalGeneration


def _is_qwen3_vl_arch(model_id: str) -> bool:
    try:
        return _resolve_vl_model_class(model_id).__name__ == "Qwen3VLForConditionalGeneration"
    except Exception:
        return False


class Qwen2VLBackend(VLMBackend):
    """Loads Qwen2 / Qwen2.5 / Qwen3 VL checkpoints based on ``model_id``."""

    name = "qwen2_vl"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        attn_implementation: Optional[str] = None,
        # Explicit: ``images`` (default) = separate image tokens; ``video`` = Qwen3 cookbook frame-list + timestamps.
        qwen_frame_input: Optional[str] = None,
        use_qwen_video_input: Optional[bool] = None,
        video_sample_fps: float = 1.0,
        qwen_video_patch_size: int = 16,
        video_total_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        image_min_pixels: Optional[int] = None,
        high_detail_pixels: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype_name = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.qwen_frame_input = _normalize_qwen_frame_input(
            qwen_frame_input, legacy_use_video=use_qwen_video_input
        )
        self.video_sample_fps = float(video_sample_fps)
        self.qwen_video_patch_size = int(qwen_video_patch_size)
        self.video_total_pixels = video_total_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        hd = bool(high_detail_pixels)
        self.image_max_pixels = (
            int(image_max_pixels)
            if image_max_pixels is not None
            else (_DEFAULT_IMAGE_MAX_PIXELS_HIGH_DETAIL if hd else None)
        )
        self.image_min_pixels = (
            int(image_min_pixels)
            if image_min_pixels is not None
            else (_DEFAULT_IMAGE_MIN_PIXELS_HIGH_DETAIL if hd else None)
        )
        self._model = None
        self._processor = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "qwen2_vl / qwen3_vl backend requires `transformers` and `torch`. "
                "Qwen3-VL needs a recent transformers (e.g. >=4.57). Use backend=mock if unsupported."
            ) from e

        try:
            _VLModel = _resolve_vl_model_class(self.model_id)
        except ImportError as e:
            raise ImportError(
                f"Could not import the VL model class for {self.model_id!r}. "
                "Upgrade transformers or use backend=mock."
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        td = dtype_map.get(self.torch_dtype_name, torch.bfloat16)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        load_kw: Dict[str, Any] = dict(
            torch_dtype=td,
            device_map=None,
            trust_remote_code=True,
        )
        if self.attn_implementation:
            load_kw["attn_implementation"] = self.attn_implementation
        self._model = _VLModel.from_pretrained(self.model_id, **load_kw)
        self._model.to(self.device)
        self._model.eval()

    def _describe_slice_qwen3_video_cookbook(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        """Match Qwen3-VL video cookbook: frame list + sample_fps + timestamped video tokens."""
        import torch

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen_frame_input='video' requires the `qwen-vl-utils` package. "
                "Install with: pip install qwen-vl-utils"
            ) from e

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        frames = list(images)
        video_ele: Dict[str, Any] = {
            "video": frames,
            "sample_fps": self.video_sample_fps,
            "raw_fps": self.video_sample_fps,
        }
        if self.video_min_pixels is not None:
            video_ele["min_pixels"] = int(self.video_min_pixels)
        if self.video_max_pixels is not None:
            video_ele["max_pixels"] = int(self.video_max_pixels)
        if self.video_total_pixels is not None:
            video_ele["total_pixels"] = int(self.video_total_pixels)

        user_content: List[Dict[str, Any]] = [
            video_ele,
            {"type": "text", "text": user_prompt},
        ]
        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=self.qwen_video_patch_size,
            return_video_metadata=True,
        )
        video_metadatas: Optional[List[Any]] = None
        if video_inputs is not None:
            v_list: List[Any] = []
            m_list: List[Any] = []
            for x in video_inputs:
                if isinstance(x, tuple) and len(x) == 2:
                    v_list.append(x[0])
                    m_list.append(x[1])
                else:
                    v_list.append(x)
            video_inputs = v_list
            video_metadatas = m_list if m_list else None
        if video_inputs is None:
            raise RuntimeError(
                "qwen_vl_utils.process_vision_info returned no video inputs for a video message — "
                "check message format and qwen-vl-utils version."
            )
        if not video_metadatas:
            raise RuntimeError(
                "Missing video_metadata from qwen_vl_utils (return_video_metadata=True). "
                "Upgrade qwen-vl-utils or file an issue with versions."
            )

        proc_kw: Dict[str, Any] = {
            **video_kwargs,
            "do_resize": False,
            "return_tensors": "pt",
            "padding": True,
        }
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **proc_kw,
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return decoded.strip()

    def describe_slice(
        self,
        images: Sequence[Image.Image],
        *,
        system_prompt: Optional[str],
        user_prompt: str,
    ) -> str:
        import torch

        if self.qwen_frame_input == "video":
            if not _is_qwen3_vl_arch(self.model_id):
                raise ValueError(
                    f"qwen_frame_input='video' only applies to Qwen3-VL–family checkpoints; "
                    f"model_id={self.model_id!r} resolves to a different architecture."
                )
            if len(images) == 0:
                raise ValueError("qwen_frame_input='video' requires at least one frame.")
            return self._describe_slice_qwen3_video_cookbook(
                images,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        content: list = []
        for im in images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": user_prompt})

        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        proc_extra: Dict[str, Any] = {}
        if self.image_max_pixels is not None or self.image_min_pixels is not None:
            ik: Dict[str, Any] = {}
            if self.image_min_pixels is not None:
                ik["min_pixels"] = int(self.image_min_pixels)
            if self.image_max_pixels is not None:
                ik["max_pixels"] = int(self.image_max_pixels)
            proc_extra["images_kwargs"] = ik
        inputs = self._processor(
            text=[text],
            images=list(images),
            return_tensors="pt",
            padding=True,
            **proc_extra,
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return decoded.strip()

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

        user_content = [{"type": "text", "text": user_prompt}]
        messages: list = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return decoded.strip()


def build_qwen2_vl_backend(params: Optional[Dict[str, Any]] = None) -> Qwen2VLBackend:
    return Qwen2VLBackend(**(params or {}))
