"""Qwen *-VL backends: Qwen2-VL, Qwen2.5-VL, Qwen3-VL (lazy transformers imports).

Registry names ``qwen2_vl`` and ``qwen3_vl`` both use this module; the HF class is
picked from ``model_id`` (e.g. ``Qwen/Qwen3-VL-4B-Instruct``)."""

from __future__ import annotations

import base64
import io
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from PIL import Image

from policy_doctor.vlm.backends.base import (
    AssistantTurn,
    TokenUsage,
    ToolCall,
    VLMBackend,
)

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
        # 4-bit / 8-bit weight loading via bitsandbytes. Use load_in_4bit for
        # the largest checkpoints (e.g. Qwen3-VL-32B fits in ~22GB across two
        # GPUs in NF4 + double-quant). When device_map is set, the model is
        # placed by accelerate; we skip the .to(device) call.
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[Any] = None,
        max_memory: Optional[Dict[Any, str]] = None,
        **_kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype_name = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.load_in_4bit = bool(load_in_4bit)
        self.load_in_8bit = bool(load_in_8bit)
        self.device_map = device_map
        self.max_memory = max_memory
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
            device_map=self.device_map,
            trust_remote_code=True,
        )
        if self.attn_implementation:
            load_kw["attn_implementation"] = self.attn_implementation
        if self.max_memory is not None:
            load_kw["max_memory"] = self.max_memory
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError(
                    "load_in_4bit/8bit requires `bitsandbytes` and a recent "
                    "`transformers` (`pip install bitsandbytes`)."
                ) from e
            if self.load_in_4bit:
                load_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=td,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            # bitsandbytes-quantized weights cannot be moved with .to(); the
            # model must be placed via device_map. Force "auto" if the caller
            # didn't pick one, so the model lands on whatever GPUs are free.
            if load_kw["device_map"] is None:
                load_kw["device_map"] = "auto"
        self._model = _VLModel.from_pretrained(self.model_id, **load_kw)
        if load_kw.get("device_map") is None:
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

    def classify_slice(
        self,
        *,
        query_images: Sequence[Image.Image],
        example_sets: Sequence[Tuple[str, Sequence[Image.Image]]],
        system_prompt: Optional[str],
        user_preamble: str,
        user_prompt: str,
        query_extra_text: Optional[str] = None,
        example_extra_texts: Optional[
            Sequence[Optional[Sequence[Optional[str]]]]
        ] = None,
    ) -> str:
        import torch

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        all_images: List[Image.Image] = []
        content: list = []
        if user_preamble:
            content.append({"type": "text", "text": user_preamble})
        for ci, (label, imgs) in enumerate(example_sets):
            content.append({"type": "text", "text": f"{label}:"})
            extras_for_group = (
                example_extra_texts[ci]
                if example_extra_texts is not None and ci < len(example_extra_texts)
                else None
            )
            for j, im in enumerate(imgs):
                content.append({"type": "image", "image": im})
                all_images.append(im)
                if extras_for_group is not None and j < len(extras_for_group):
                    extra = extras_for_group[j]
                    if extra:
                        content.append({"type": "text", "text": extra})
        content.append({"type": "text", "text": "Query:"})
        for im in query_images:
            content.append({"type": "image", "image": im})
            all_images.append(im)
        if query_extra_text:
            content.append({"type": "text", "text": query_extra_text})
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
            images=all_images,
            return_tensors="pt",
            padding=True,
            **proc_extra,
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return decoded.strip()

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


    # ------------------------------------------------------------------
    # Tool-use primitive (used by the agentic proposal loop)
    # ------------------------------------------------------------------

    _TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
    )

    def chat_with_tools(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        seed: Optional[int] = None,
    ) -> AssistantTurn:
        """One agentic turn against a Qwen *-VL checkpoint (Hermes-style tools).

        Implementation strategy:
          * Convert provider-neutral (Anthropic-shaped) messages into Qwen
            chat-template messages: ``user`` text/image blocks pass through;
            ``assistant`` ``tool_use`` blocks are serialised as Hermes-style
            ``<tool_call>{json}</tool_call>`` text inside the assistant turn;
            ``tool_result`` blocks become ``role=tool`` text messages, with
            any inline images promoted into a follow-up ``user`` content
            block (Qwen's tool turn does not accept images).
          * Hand tool definitions to the chat template via ``tools=[...]``;
            the template injects them into the system prompt in the format
            Qwen expects (and that Qwen3-VL is trained to emit
            ``<tool_call>`` blocks for).
          * Decode the new tokens with ``skip_special_tokens=False`` so the
            ``<tool_call>`` tags survive ``batch_decode``, then parse them.
        """
        import torch

        self._lazy_init()
        assert self._processor is not None and self._model is not None

        qwen_msgs, all_images = self._convert_messages_for_qwen(messages, system)
        qwen_tools = [self._tool_to_qwen(t) for t in tools]

        text = self._processor.apply_chat_template(
            qwen_msgs,
            tools=qwen_tools or None,
            add_generation_prompt=True,
            tokenize=False,
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
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
            **proc_extra,
        )

        # Place inputs on the device that holds the model's first parameter.
        # When device_map is set (e.g. "auto" for multi-GPU 4-bit), accelerate
        # handles the rest via hooks; we just need the input tensors to start
        # on the correct GPU.
        target_dev = (
            next(self._model.parameters()).device
            if self.device_map is not None
            else self.device
        )
        inputs = inputs.to(target_dev)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_tokens)}
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False
        if seed is not None:
            torch.manual_seed(int(seed))

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)

        in_len = int(inputs["input_ids"].shape[1])
        trimmed = out_ids[:, in_len:]
        # skip_special_tokens=False so <tool_call> XML tags survive — they
        # are treated as ordinary text by Qwen tokenizers, but some
        # downstream wrappers strip leading/trailing specials regardless.
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=False)[0]
        text_out, tool_calls = self._parse_qwen_assistant(decoded)

        return AssistantTurn(
            text=text_out,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            usage=TokenUsage(
                input_tokens=in_len,
                output_tokens=int(trimmed.shape[1]),
            ),
            raw=decoded,
        )

    # ---- chat_with_tools helpers --------------------------------------

    @staticmethod
    def _tool_to_qwen(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Provider-neutral tool dict -> Qwen ``{"type":"function","function":...}``."""
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {
                    "type": "object", "properties": {},
                },
            },
        }

    @staticmethod
    def _extract_image_from_block(blk: Dict[str, Any]) -> Optional[Image.Image]:
        """Decode an Anthropic-shaped image block into a PIL.Image.

        Anthropic ships images as
          ``{"type": "image", "source": {"type": "base64", "media_type": ..., "data": <b64>}}``.
        We accept that, and as a convenience also accept a direct PIL handle
        under ``{"image": <PIL>}``.
        """
        src = blk.get("source")
        if isinstance(src, dict) and src.get("type") == "base64":
            data = base64.b64decode(src.get("data", ""))
            return Image.open(io.BytesIO(data)).convert("RGB")
        img = blk.get("image")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        return None

    def _convert_messages_for_qwen(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Translate Anthropic-shaped messages into Qwen chat-template messages.

        Returns ``(qwen_messages, ordered_images)`` where the image list is
        flattened in the same order the chat template will consume them.
        """
        out: List[Dict[str, Any]] = []
        all_images: List[Image.Image] = []
        if system:
            out.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            if role == "assistant":
                parts: List[str] = []
                for blk in content:
                    btype = blk.get("type")
                    if btype == "text" and blk.get("text"):
                        parts.append(blk["text"])
                    elif btype == "tool_use":
                        tc_json = json.dumps(
                            {"name": blk.get("name", ""),
                             "arguments": blk.get("input") or {}},
                            ensure_ascii=False,
                        )
                        parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")
                out.append({
                    "role": "assistant",
                    "content": "\n".join(parts) if parts else "",
                })
                continue

            # role == "user" — may carry tool_results, text, or images.
            user_blocks: List[Dict[str, Any]] = []
            for blk in content:
                btype = blk.get("type")
                if btype == "tool_result":
                    inner = blk.get("content") or []
                    text_parts: List[str] = []
                    inner_images: List[Image.Image] = []
                    for ib in inner:
                        if ib.get("type") == "text":
                            txt = ib.get("text", "")
                            if txt:
                                text_parts.append(txt)
                        elif ib.get("type") == "image":
                            img = self._extract_image_from_block(ib)
                            if img is not None:
                                inner_images.append(img)
                    text_combined = "\n".join(text_parts).strip()
                    if blk.get("is_error"):
                        text_combined = (
                            f"[error]\n{text_combined}" if text_combined else "[error]"
                        )
                    if not text_combined and not inner_images:
                        text_combined = "(empty)"
                    out.append({
                        "role": "tool",
                        "content": text_combined or "(see attached image)",
                    })
                    if inner_images:
                        # Qwen's tool role does not accept images. Promote
                        # them into a follow-up user message so the chat
                        # template can place <|image_pad|> tokens for the
                        # processor to fill from the images list.
                        img_blocks = [
                            {"type": "image", "image": img} for img in inner_images
                        ]
                        out.append({"role": "user", "content": img_blocks})
                        all_images.extend(inner_images)
                elif btype == "text":
                    user_blocks.append({"type": "text", "text": blk.get("text", "")})
                elif btype == "image":
                    img = self._extract_image_from_block(blk)
                    if img is not None:
                        user_blocks.append({"type": "image", "image": img})
                        all_images.append(img)
            if user_blocks:
                out.append({"role": "user", "content": user_blocks})

        return out, all_images

    @classmethod
    def _parse_qwen_assistant(
        cls, decoded: str,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Pull ``<tool_call>{...}</tool_call>`` blocks out of *decoded*.

        Returns ``(text_remainder, tool_calls)``. Strips chat-template special
        markers some decoders leave when special tokens are not skipped.
        Malformed JSON inside a tool_call falls through as plain text rather
        than killing the turn.
        """
        cleaned = decoded
        for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            cleaned = cleaned.replace(tok, "")

        tool_calls: List[ToolCall] = []
        parts: List[str] = []
        last_end = 0
        for m in cls._TOOL_CALL_RE.finditer(cleaned):
            parts.append(cleaned[last_end:m.start()])
            try:
                obj = json.loads(m.group(1))
                args = obj.get("arguments")
                if args is None:
                    args = obj.get("parameters") or {}
                tool_calls.append(ToolCall(
                    id=f"qwen_{uuid.uuid4().hex[:12]}",
                    name=str(obj.get("name", "")),
                    arguments=dict(args) if isinstance(args, dict) else {},
                ))
            except (json.JSONDecodeError, TypeError, ValueError):
                parts.append(m.group(0))
            last_end = m.end()
        parts.append(cleaned[last_end:])
        text = "".join(parts).strip()
        return (text or None, tool_calls)


def build_qwen2_vl_backend(params: Optional[Dict[str, Any]] = None) -> Qwen2VLBackend:
    return Qwen2VLBackend(**(params or {}))
