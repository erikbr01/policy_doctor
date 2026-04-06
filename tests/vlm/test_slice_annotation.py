"""Tests for VLM slice annotation (mock backend, frame extraction)."""

from __future__ import annotations

import json
import pathlib
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from policy_doctor.vlm.annotate import (
    eval_dirs_equivalent,
    resolve_source_eval_dir_for_jsonl,
    run_slice_annotation_for_eval,
)
from policy_doctor.vlm.debug_plots import save_slice_annotation_debug_png
from policy_doctor.vlm.backends.mock import MockVLMBackend
from policy_doctor.vlm.prompts import (
    extract_slice_final_label,
    normalize_slice_reasoning_effort,
)
from policy_doctor.vlm.behavior_summarize import (
    group_slice_labels_by_cluster,
    run_behavior_summarization,
)
from policy_doctor.vlm.frames import extract_window_frames, resolve_window_indices
from policy_doctor.vlm.prompts import resolve_task_vlm_prompts_path
from policy_doctor.plotting.vlm_montage import create_scrollable_frame_strip_html
from policy_doctor.vlm.registry import get_vlm_backend, list_vlm_backend_names


class TestSliceAnnotation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.eval_dir = self.root / "eval"
        ep_dir = self.eval_dir / "episodes"
        ep_dir.mkdir(parents=True)

        rows = []
        for t in range(10):
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img[..., 0] = t * 20
            rows.append({"timestep": t, "img": img})
        df = pd.DataFrame(rows)
        with open(ep_dir / "ep0000_succ.pkl", "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(ep_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump({"length": 10, "episode_lengths": [10], "episode_successes": [True]}, f)

        self.cluster_dir = self.root / "clust"
        self.cluster_dir.mkdir()
        labels = np.array([0, 1, 2], dtype=np.int64)
        np.save(self.cluster_dir / "cluster_labels.npy", labels)
        meta = [
            {"rollout_idx": 0, "window_start": 0, "window_end": 3},
            {"rollout_idx": 0, "window_start": 2, "window_end": 5},
            {"rollout_idx": 0, "start": 1, "end": 4},
        ]
        with open(self.cluster_dir / "metadata.json", "w") as f:
            json.dump(meta, f)
        with open(self.cluster_dir / "manifest.yaml", "w") as f:
            yaml.safe_dump({"level": "rollout", "n_samples": 3}, f)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_resolve_window_indices(self) -> None:
        r, a, b = resolve_window_indices({"rollout_idx": 1, "window_start": 2, "window_end": 7})
        self.assertEqual((r, a, b), (1, 2, 7))
        r2, a2, b2 = resolve_window_indices({"rollout_idx": 0, "start": 1, "end": 3})
        self.assertEqual((r2, a2, b2), (0, 1, 3))

    def test_extract_window_frames(self) -> None:
        rng = np.random.default_rng(0)
        imgs = extract_window_frames(
            self.eval_dir, 0, 0, 5, max_frames=3, rng=rng
        )
        self.assertEqual(len(imgs), 3)

    def test_extract_window_frames_all_timesteps_when_max_none(self) -> None:
        rng = np.random.default_rng(0)
        imgs = extract_window_frames(
            self.eval_dir, 0, 0, 5, max_frames=None, rng=rng
        )
        self.assertEqual(len(imgs), 5)

    def test_scrollable_frame_strip_html(self) -> None:
        from PIL import Image as PILImage

        html = create_scrollable_frame_strip_html(
            [PILImage.new("RGB", (8, 6)), PILImage.new("RGB", (8, 6))]
        )
        self.assertIn("overflow-x:auto", html)
        self.assertEqual(html.count("data:image/png;base64,"), 2)

    def test_max_frames_none_loads_full_window(self) -> None:
        be = MockVLMBackend()
        records, _ = run_slice_annotation_for_eval(
            eval_dir=self.eval_dir,
            clustering_dir=self.cluster_dir,
            backend=be,
            task_hint="test_task",
            prompts_file=None,
            prompts_inline=None,
            repo_root=None,
            max_slices=10,
            max_frames_per_slice=None,
            random_seed=0,
        )
        self.assertIn("frames=3", records[0]["label"])

    def test_reasoning_high_extracts_final_line(self) -> None:
        class _CoTMock(MockVLMBackend):
            def describe_slice(self, images, *, system_prompt, user_prompt):
                return "thinking...\nFINAL: compact caption"

        records, _ = run_slice_annotation_for_eval(
            eval_dir=self.eval_dir,
            clustering_dir=self.cluster_dir,
            backend=_CoTMock(),
            task_hint="test_task",
            prompts_file=None,
            prompts_inline={"reasoning_effort": "high"},
            repo_root=None,
            max_slices=10,
            max_frames_per_slice=2,
            random_seed=0,
        )
        self.assertEqual(records[0]["label"], "compact caption")
        self.assertIn("raw_model_output", records[0])

    def test_extract_final_label_helpers(self) -> None:
        self.assertEqual(normalize_slice_reasoning_effort("HIGH"), "high")
        self.assertEqual(
            extract_slice_final_label("a\nFINAL: x", reasoning_effort="high"), "x"
        )
        self.assertEqual(
            extract_slice_final_label("no final", reasoning_effort="high"), "no final"
        )
        with self.assertRaises(ValueError):
            normalize_slice_reasoning_effort("bogus")

    def test_run_slice_annotation_mock(self) -> None:
        be = MockVLMBackend()
        records, pver = run_slice_annotation_for_eval(
            eval_dir=self.eval_dir,
            clustering_dir=self.cluster_dir,
            backend=be,
            task_hint="test_task",
            prompts_file=None,
            prompts_inline=None,
            repo_root=None,
            max_slices=10,
            max_frames_per_slice=2,
            random_seed=0,
        )
        self.assertEqual(len(records), 3)
        self.assertTrue(all("label" in r for r in records))
        self.assertEqual(records[0]["rollout_idx"], 0)
        self.assertTrue(len(pver) == 16)
        self.assertFalse(any("debug_plot_path" in r for r in records))

    def test_save_slice_annotation_debug_png(self) -> None:
        from PIL import Image as PILImage

        out = self.root / "slice_debug.png"
        save_slice_annotation_debug_png(
            out,
            images=[PILImage.new("RGB", (12, 10))],
            system_prompt="system line",
            user_prompt="user prompt body",
            model_label="mock caption",
            meta={
                "slice_index": 7,
                "rollout_idx": 0,
                "window_start": 0,
                "window_end": 3,
                "cluster_id": 2,
                "backend": "mock",
            },
        )
        self.assertTrue(out.is_file())
        self.assertGreater(out.stat().st_size, 500)

    def test_run_slice_annotation_writes_debug_plots(self) -> None:
        be = MockVLMBackend()
        ddir = self.root / "debug_plots_out"
        records, pver = run_slice_annotation_for_eval(
            eval_dir=self.eval_dir,
            clustering_dir=self.cluster_dir,
            backend=be,
            task_hint="test_task",
            prompts_file=None,
            prompts_inline=None,
            repo_root=None,
            max_slices=10,
            max_frames_per_slice=2,
            random_seed=0,
            debug_plots_dir=ddir,
        )
        self.assertEqual(len(records), 3)
        for r in records:
            p = r.get("debug_plot_path")
            self.assertIsInstance(p, str)
            self.assertTrue(pathlib.Path(p).is_file())

    def test_registry_mock(self) -> None:
        names = list_vlm_backend_names()
        self.assertIn("mock", names)
        self.assertIn("molmo", names)
        self.assertIn("molmo2", names)
        b = get_vlm_backend("mock", {})
        self.assertIsInstance(b, MockVLMBackend)

    def test_group_slice_labels_by_cluster(self) -> None:
        recs = [
            {"cluster_id": 1, "label": "a"},
            {"cluster_id": 0, "label": "b"},
            {"cluster_id": 1, "label": "c"},
        ]
        g = group_slice_labels_by_cluster(recs, max_labels_per_cluster=None)
        self.assertEqual(g[0], ["b"])
        self.assertEqual(g[1], ["a", "c"])

    def test_resolve_source_eval_dir_from_records(self) -> None:
        p = pathlib.Path("/tmp/fake/annotations_seed0.jsonl")
        recs = [{"source_eval_dir": "/data/eval/a"}]
        self.assertEqual(resolve_source_eval_dir_for_jsonl(p, recs), "/data/eval/a")

    def test_eval_dirs_equivalent(self) -> None:
        self.assertTrue(
            eval_dirs_equivalent(pathlib.Path("/a/../a"), pathlib.Path("/a"))
        )

    def test_resolve_task_vlm_prompts_transport_mh(self) -> None:
        p = resolve_task_vlm_prompts_path("transport_mh_mar27")
        self.assertIsNotNone(p)
        assert p is not None
        self.assertEqual(p.name, "transport_mh.yaml")
        self.assertIsNone(resolve_task_vlm_prompts_path("unknown_task_xyz"))

    def test_run_behavior_summarization_mock(self) -> None:
        recs = [
            {"cluster_id": 0, "label": "grasp cup"},
            {"cluster_id": 0, "label": "move arm"},
            {"cluster_id": 1, "label": "idle"},
        ]
        be = MockVLMBackend()
        sums, pver = run_behavior_summarization(
            recs,
            backend=be,
            task_hint="t",
            prompts_file=None,
            prompts_inline=None,
            repo_root=None,
            max_slice_labels_per_cluster=None,
            max_clusters=None,
        )
        self.assertEqual(len(sums), 2)
        self.assertEqual(sums[0]["cluster_id"], 0)
        self.assertIn("behavior cluster=0", sums[0]["summary"])
        self.assertTrue(len(pver) == 16)


if __name__ == "__main__":
    unittest.main()
