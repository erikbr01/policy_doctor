"""Unit tests for the Gemini VLM backend (mocked — no real API calls)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _make_mock_genai() -> MagicMock:
    """Return a fresh mock that looks like google.generativeai."""
    mock_response = MagicMock()
    mock_response.text = "robot grasps the cube"
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    mock_genai.types.GenerationConfig.return_value = MagicMock()
    return mock_genai


class TestGeminiRegistry(unittest.TestCase):
    def test_registry_has_gemini(self):
        from policy_doctor.vlm.registry import list_vlm_backend_names

        names = list_vlm_backend_names()
        self.assertIn("gemini", names)
        self.assertIn("gemini_flash", names)


class TestGeminiBackendMocked(unittest.TestCase):
    """Verify GeminiVLMBackend behaviour using a mocked google.generativeai module."""

    def _make_backend(self, mock_genai):
        """Construct a GeminiVLMBackend with _require_genai patched."""
        from policy_doctor.vlm.backends.gemini import GeminiVLMBackend

        with patch(
            "policy_doctor.vlm.backends.gemini._require_genai", return_value=mock_genai
        ):
            backend = GeminiVLMBackend(model_name="gemini-2.0-flash", api_key="fake_key")
        return backend

    def test_describe_slice_calls_generate_content(self):
        mock_genai = _make_mock_genai()
        backend = self._make_backend(mock_genai)

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (64, 64), color=(128, 64, 32))
        result = backend.describe_slice(
            [img],
            system_prompt="You are a robot observer.",
            user_prompt="Describe the robot action.",
        )
        self.assertEqual(result, "robot grasps the cube")
        backend._model.generate_content.assert_called_once()

    def test_describe_slice_no_system_prompt(self):
        mock_genai = _make_mock_genai()
        backend = self._make_backend(mock_genai)
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (32, 32))
        result = backend.describe_slice([img], system_prompt=None, user_prompt="What happens?")
        self.assertIsInstance(result, str)

    def test_summarize_behavior_labels(self):
        mock_genai = _make_mock_genai()
        backend = self._make_backend(mock_genai)
        result = backend.summarize_behavior_labels(
            cluster_id=0,
            slice_labels=["grasp", "pick", "lift"],
            task_hint="square manipulation",
            system_prompt=None,
            user_prompt="Summarize these behaviors.",
        )
        self.assertIsInstance(result, str)
        backend._model.generate_content.assert_called_once()

    def test_evaluate_slice_caption_coherency(self):
        mock_genai = _make_mock_genai()
        mock_genai.GenerativeModel.return_value.generate_content.return_value.text = (
            '{"coherent": true, "score": 0.9, "rationale": "all consistent"}'
        )
        backend = self._make_backend(mock_genai)
        result = backend.evaluate_slice_caption_coherency(
            cluster_id=0,
            slice_labels=["grasp", "pick", "lift"],
            task_hint="square manipulation",
            system_prompt=None,
            user_prompt="Are these coherent?",
        )
        self.assertIsInstance(result, str)
        self.assertIn("coherent", result)

    def test_import_error_without_package(self):
        """_require_genai raises ImportError when google.generativeai is absent."""
        import sys

        # Remove from cache so import fails
        saved = {k: v for k, v in sys.modules.items() if "google" in k}
        for k in list(saved.keys()):
            sys.modules.pop(k, None)
        try:
            from policy_doctor.vlm.backends.gemini import _require_genai

            with self.assertRaises(ImportError):
                _require_genai()
        finally:
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
