"""Unit tests: each VLM backend's load path + one inference call.

Heavy Hugging Face weights are never downloaded — ``from_pretrained`` and ``generate``
are mocked. Requires ``torch``, ``transformers``, and ``PIL`` (same as production VLM).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from policy_doctor.vlm.backends.molmo2 import Molmo2Backend, build_molmo_backend, build_molmo2_backend
from policy_doctor.vlm.backends.mock import MockVLMBackend
from policy_doctor.vlm.backends.qwen2_vl import (
    Qwen2VLBackend,
    _normalize_qwen_frame_input,
    _resolve_vl_model_class,
)
from policy_doctor.vlm.registry import get_vlm_backend, list_vlm_backend_names


def _tiny_rgb() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(10, 20, 30))


class _QwenLikeBatch(dict):
    """Minimal stand-in for HF processor output supporting ``.to(device)`` and ``**inputs``."""

    def to(self, device):  # noqa: ARG002
        return self


class TestVLMBackendRegistry(unittest.TestCase):
    def test_list_includes_all_backends(self) -> None:
        names = list_vlm_backend_names()
        for n in ("mock", "qwen2_vl", "qwen3_vl", "cosmos_reason2", "molmo", "molmo2"):
            with self.subTest(n=n):
                self.assertIn(n, names)

    def test_mock_factory(self) -> None:
        b = get_vlm_backend("mock", {})
        self.assertIsInstance(b, MockVLMBackend)

    def test_molmo_alias_sets_name(self) -> None:
        b2 = build_molmo2_backend({})
        bm = build_molmo_backend({})
        self.assertEqual(b2.name, "molmo2")
        self.assertEqual(bm.name, "molmo")
        self.assertIsInstance(b2, Molmo2Backend)
        self.assertIsInstance(bm, Molmo2Backend)

    def test_qwen3_overrides_backend_name(self) -> None:
        b = get_vlm_backend("qwen3_vl", {"model_id": "Qwen/Qwen3-VL-4B-Instruct", "device": "cpu"})
        self.assertIsInstance(b, Qwen2VLBackend)
        self.assertEqual(b.name, "qwen3_vl")

    def test_cosmos_reason2_factory_defaults(self) -> None:
        b = get_vlm_backend("cosmos_reason2", {"device": "cpu"})
        self.assertIsInstance(b, Qwen2VLBackend)
        self.assertEqual(b.name, "cosmos_reason2")
        self.assertIn("Cosmos-Reason2", b.model_id)

    def test_resolve_cosmos_uses_qwen3_vl_class(self) -> None:
        from transformers import Qwen3VLForConditionalGeneration

        cls = _resolve_vl_model_class("nvidia/Cosmos-Reason2-8B")
        self.assertIs(cls, Qwen3VLForConditionalGeneration)


class TestQwenFrameInputAndPixels(unittest.TestCase):
    def test_normalize_qwen_frame_input_aliases(self) -> None:
        self.assertEqual(
            _normalize_qwen_frame_input("multi-image", legacy_use_video=None), "images"
        )
        self.assertEqual(
            _normalize_qwen_frame_input("frame_list", legacy_use_video=None), "video"
        )
        self.assertEqual(_normalize_qwen_frame_input(None, legacy_use_video=True), "video")
        self.assertEqual(_normalize_qwen_frame_input(None, legacy_use_video=False), "images")

    def test_normalize_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_qwen_frame_input("bogus", legacy_use_video=None)

    def test_legacy_use_qwen_video_input_maps_when_unset(self) -> None:
        b = Qwen2VLBackend(
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            device="cpu",
            use_qwen_video_input=True,
        )
        self.assertEqual(b.qwen_frame_input, "video")
        b_explicit = Qwen2VLBackend(
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            device="cpu",
            qwen_frame_input="images",
            use_qwen_video_input=True,
        )
        self.assertEqual(b_explicit.qwen_frame_input, "images")
        b2 = Qwen2VLBackend(
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            device="cpu",
            use_qwen_video_input=False,
        )
        self.assertEqual(b2.qwen_frame_input, "images")

    def test_high_detail_sets_image_pixel_caps(self) -> None:
        b = Qwen2VLBackend(
            model_id="Qwen/Qwen3-VL-4B-Instruct",
            device="cpu",
            high_detail_pixels=True,
        )
        self.assertIsNotNone(b.image_max_pixels)
        self.assertIsNotNone(b.image_min_pixels)

    def test_qwen2_checkpoint_rejects_video_mode_before_load(self) -> None:
        b = Qwen2VLBackend(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            device="cpu",
            qwen_frame_input="video",
        )
        with self.assertRaises(ValueError):
            b.describe_slice([_tiny_rgb()], system_prompt=None, user_prompt="x")


class TestMockBackendInference(unittest.TestCase):
    def test_describe_and_summarize(self) -> None:
        b = get_vlm_backend("mock", {})
        out = b.describe_slice([_tiny_rgb()], system_prompt="sys", user_prompt="user")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)
        s = b.summarize_behavior_labels(
            cluster_id=0,
            slice_labels=["a", "b"],
            task_hint="t",
            system_prompt=None,
            user_prompt="summarize",
        )
        self.assertIsInstance(s, str)


@patch("policy_doctor.vlm.backends.qwen2_vl._resolve_vl_model_class")
@patch("transformers.AutoProcessor.from_pretrained")
class TestQwenBackendsMockedLoad(unittest.TestCase):
    """Qwen2-VL and Qwen3-VL share ``qwen2_vl.py``; only registry name differs for qwen3."""

    def _setup_qwen_mocks(self, m_proc_fp: MagicMock, m_resolve: MagicMock) -> tuple[MagicMock, MagicMock]:
        mock_vl_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_vl_cls.from_pretrained.return_value = mock_model
        m_resolve.return_value = mock_vl_cls

        in_len = 3
        input_ids = torch.ones((1, in_len), dtype=torch.long)
        batch = _QwenLikeBatch({"input_ids": input_ids})
        fake_proc = MagicMock()
        fake_proc.apply_chat_template.return_value = "<<prompt>>"
        fake_proc.side_effect = lambda **kwargs: batch
        fake_proc.batch_decode = MagicMock(return_value=["  generated caption  "])
        m_proc_fp.return_value = fake_proc

        total_len = in_len + 2
        mock_model.generate.return_value = torch.ones((1, total_len), dtype=torch.long)
        return mock_model, mock_vl_cls

    def test_qwen2_vl_load_and_describe_slice(
        self, m_proc_fp: MagicMock, m_resolve: MagicMock
    ) -> None:
        mock_model, mock_vl_cls = self._setup_qwen_mocks(m_proc_fp, m_resolve)
        b = get_vlm_backend(
            "qwen2_vl",
            {"model_id": "Qwen/Qwen2-VL-2B-Instruct", "device": "cpu"},
        )
        out = b.describe_slice([_tiny_rgb()], system_prompt=None, user_prompt="What happens?")
        self.assertEqual(out, "generated caption")
        mock_vl_cls.from_pretrained.assert_called_once()
        m_proc_fp.assert_called_once()
        mock_model.generate.assert_called_once()

    def test_qwen3_vl_load_and_describe_slice(
        self, m_proc_fp: MagicMock, m_resolve: MagicMock
    ) -> None:
        self._setup_qwen_mocks(m_proc_fp, m_resolve)
        b = get_vlm_backend(
            "qwen3_vl",
            {"model_id": "Qwen/Qwen3-VL-4B-Instruct", "device": "cpu"},
        )
        out = b.describe_slice([_tiny_rgb()], system_prompt="Be brief.", user_prompt="Q")
        self.assertEqual(out, "generated caption")

    def test_qwen3_vl_summarize_behavior_text_only(
        self, m_proc_fp: MagicMock, m_resolve: MagicMock
    ) -> None:
        self._setup_qwen_mocks(m_proc_fp, m_resolve)
        b = get_vlm_backend("qwen3_vl", {"model_id": "Qwen/Qwen3-VL-4B-Instruct", "device": "cpu"})
        s = b.summarize_behavior_labels(
            cluster_id=1,
            slice_labels=["x"],
            task_hint="t",
            system_prompt="sys",
            user_prompt="User text",
        )
        self.assertEqual(s, "generated caption")


@patch("transformers.AutoModelForImageTextToText.from_pretrained")
@patch("transformers.AutoProcessor.from_pretrained")
class TestMolmo2BackendMockedLoad(unittest.TestCase):
    def _setup_molmo_mocks(self, m_proc_fp: MagicMock, m_model_fp: MagicMock) -> MagicMock:
        in_len = 4
        input_ids = torch.ones((1, in_len), dtype=torch.long)
        tmpl_out = {"input_ids": input_ids, "attention_mask": torch.ones((1, in_len), dtype=torch.long)}

        fake_tok = MagicMock()
        fake_tok.decode.return_value = "  molmo output  "

        fake_proc = MagicMock()
        fake_proc.apply_chat_template.return_value = tmpl_out
        fake_proc.tokenizer = fake_tok
        m_proc_fp.return_value = fake_proc

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.device = torch.device("cpu")
        out_len = in_len + 3
        mock_model.generate.return_value = torch.ones((1, out_len), dtype=torch.long)
        m_model_fp.return_value = mock_model
        return mock_model

    def test_molmo2_load_and_describe_slice(self, m_proc_fp: MagicMock, m_model_fp: MagicMock) -> None:
        mock_model = self._setup_molmo_mocks(m_proc_fp, m_model_fp)
        b = get_vlm_backend("molmo2", {"model_id": "allenai/Molmo2-4B", "device": "cpu"})
        out = b.describe_slice([_tiny_rgb()], system_prompt="S", user_prompt="U")
        self.assertEqual(out, "molmo output")
        m_proc_fp.assert_called_once()
        m_model_fp.assert_called_once()
        mock_model.generate.assert_called_once()

    def test_molmo_registry_alias_load(self, m_proc_fp: MagicMock, m_model_fp: MagicMock) -> None:
        self._setup_molmo_mocks(m_proc_fp, m_model_fp)
        b = get_vlm_backend("molmo", {"model_id": "allenai/Molmo2-4B", "device": "cpu"})
        self.assertEqual(b.name, "molmo")
        out = b.describe_slice([_tiny_rgb()], system_prompt=None, user_prompt="hi")
        self.assertEqual(out, "molmo output")

    def test_molmo2_summarize_behavior(self, m_proc_fp: MagicMock, m_model_fp: MagicMock) -> None:
        self._setup_molmo_mocks(m_proc_fp, m_model_fp)
        b = get_vlm_backend("molmo2", {"model_id": "allenai/Molmo2-4B", "device": "cpu"})
        s = b.summarize_behavior_labels(
            cluster_id=0,
            slice_labels=[],
            task_hint="t",
            system_prompt=None,
            user_prompt="Summarize these.",
        )
        self.assertEqual(s, "molmo output")


if __name__ == "__main__":
    unittest.main()
