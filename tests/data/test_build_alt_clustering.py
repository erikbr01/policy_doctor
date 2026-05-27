"""Smoke test: scripts/build_alt_clustering.py produces a valid E1-compatible dir.

Builds a synthetic eval directory (matching the eval_save_episodes layout),
runs the script as a subprocess, and verifies the output dir is consumable
by the existing E1 sample-plan builder.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

import policy_doctor

_REPO_ROOT = Path(policy_doctor.__file__).resolve().parent.parent
_BUILDER = _REPO_ROOT / "scripts" / "experiments" / "build_alt_clustering.py"


def _make_eval_dir(tmp: Path, n_episodes: int = 3, ep_len: int = 20) -> Path:
    eval_dir = tmp / "eval"
    ep_dir = eval_dir / "episodes"
    ep_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for ep_i in range(n_episodes):
        rows = []
        for t in range(ep_len):
            rows.append({
                "obs": rng.standard_normal((2, 7)).astype(np.float32),
                "action": rng.standard_normal((3, 5)).astype(np.float32),
                "img": np.zeros((4, 4, 3), dtype=np.uint8),
                "success": ep_i % 2 == 0,
            })
        df = pd.DataFrame(rows)
        suffix = "succ" if ep_i % 2 == 0 else "fail"
        with open(ep_dir / f"ep{ep_i:04d}_{suffix}.pkl", "wb") as f:
            pickle.dump(df, f)
    with open(ep_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump({
            "episode_lengths": [ep_len] * n_episodes,
            "episode_successes": [bool(i % 2 == 0) for i in range(n_episodes)],
        }, f)
    return eval_dir


class BuildAltClusteringSmoke(unittest.TestCase):
    def test_state_action_runs_end_to_end(self):
        with TemporaryDirectory() as td:
            tdp = Path(td)
            eval_dir = _make_eval_dir(tdp, n_episodes=3, ep_len=20)
            out_dir = tdp / "out_state_action_k3"

            res = subprocess.run([
                sys.executable, str(_BUILDER),
                "--representation", "state_action",
                "--eval_dir", str(eval_dir),
                "--out_dir", str(out_dir),
                "--window_width", "5",
                "--stride", "5",
                "--aggregation", "mean",
                "--prescale", "standard",
                "--reducer", "umap",
                "--umap_n_components", "4",
                "--n_clusters", "3",
                "--seed", "42",
            ], capture_output=True, text=True, timeout=120)

            if res.returncode != 0:
                self.fail(
                    f"build_alt_clustering exited {res.returncode}\n"
                    f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
                )

            for fname in ("manifest.yaml", "cluster_labels.npy",
                          "metadata.json", "embeddings_reduced.npy",
                          "clustering_models.pkl"):
                self.assertTrue(
                    (out_dir / fname).exists(),
                    f"missing {fname} in {out_dir}",
                )

            labels = np.load(out_dir / "cluster_labels.npy")
            embed = np.load(out_dir / "embeddings_reduced.npy")
            with open(out_dir / "metadata.json") as f:
                meta = json.load(f)
            with open(out_dir / "manifest.yaml") as f:
                manifest = yaml.safe_load(f)

            n_slices = len(labels)
            self.assertGreater(n_slices, 0)
            self.assertEqual(embed.shape[0], n_slices)
            self.assertEqual(embed.shape[1], 4)
            self.assertEqual(len(meta), n_slices)
            self.assertEqual(manifest["slice_representation"], "state_action")
            self.assertEqual(manifest["n_clusters"], 3)
            for m in meta:
                for k in ("rollout_idx", "window_start", "window_end",
                          "window_width", "success"):
                    self.assertIn(k, m)

    def test_e1_planner_consumes_output(self):
        """The planner used by E1 must accept the dir we produce."""
        from policy_doctor.vlm.cluster_classification import build_sample_plan
        from policy_doctor.vlm.annotate import load_clustering_artifacts

        with TemporaryDirectory() as td:
            tdp = Path(td)
            eval_dir = _make_eval_dir(tdp, n_episodes=4, ep_len=20)
            out_dir = tdp / "out_state_k4"

            res = subprocess.run([
                sys.executable, str(_BUILDER),
                "--representation", "state",
                "--eval_dir", str(eval_dir),
                "--out_dir", str(out_dir),
                "--window_width", "5",
                "--stride", "5",
                "--aggregation", "mean",
                "--prescale", "standard",
                "--umap_n_components", "4",
                "--n_clusters", "3",
                "--seed", "42",
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(res.returncode, 0,
                             msg=res.stdout + "\n" + res.stderr)

            labels, metadata, manifest = load_clustering_artifacts(out_dir)
            embeddings = np.load(out_dir / "embeddings_reduced.npy")
            plan = build_sample_plan(
                labels, metadata, embeddings,
                n_example=1, n_query=1,
                rng=np.random.default_rng(0),
                global_episode_disjoint=True,
            )
            self.assertIn("cluster_ids", plan)
            self.assertEqual(plan.get("global_episode_disjoint"), True)
            for cid in plan["cluster_ids"]:
                cluster = plan["clusters"][cid]
                self.assertIn("query_origins", cluster)
                self.assertIn("global_disjointness_status", cluster)


if __name__ == "__main__":
    unittest.main()
