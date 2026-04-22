"""Generate MimicGen demonstrations from a single seed trajectory — pipeline step.

This step investigates whether MimicGen-generated demonstrations are within the
original seed demonstration's state-space distribution by comparing end-effector
(EEF) trajectories.

**Auto-wiring with SelectMimicgenSeedFromGraphStep**: if
:class:`~policy_doctor.curation_pipeline.steps.select_mimicgen_seed_from_graph
.SelectMimicgenSeedFromGraphStep` has been run in the same pipeline, this step
automatically uses its materialised ``seed.hdf5`` as the seed input, skipping
the source-dataset lookup entirely.  The graph-based seed always takes
precedence when available.

Config keys (under ``mimicgen_datagen``):
    source_dataset_path  Path to the robomimic HDF5 with source demos.
                         Defaults to the standard MimicGen Square D1 path.
                         Ignored when SelectMimicgenSeedFromGraphStep is done.
    seed_demo_key        Demo to use as the seed.  Default ``"demo_0"``.
                         Set to ``"random"`` to draw randomly from available
                         ``demo_*`` keys (deterministic via ``seed_random_seed``).
                         Ignored when SelectMimicgenSeedFromGraphStep is done.
    seed_random_seed     RNG seed used when ``seed_demo_key="random"`` (default 42).
    num_trials           Number of MimicGen generation attempts (default 50).
    output_dir           Base output directory (default
                         ``"data/outputs/mimicgen_datagen"``).
    task_name            MimicGen task name (default ``"square"``).
    env_interface_name   (default ``"MG_Square"``)
    env_interface_type   (default ``"robosuite"``)

Result JSON (``step_dir/result.json``):
    seed_demo_key        The demo key that was actually used.
    seed_source          ``"graph"`` when wired from SelectMimicgenSeedFromGraphStep,
                         ``"source_dataset"`` otherwise.
    seed_eef_xyz         List of [[x,y,z], ...] for the seed demo (after prepare).
    generated_eef_xyz    List of per-demo [[x,y,z], ...] lists.
    generated_hdf5_path  Absolute path to the generated ``demo.hdf5``.
    stats                MimicGen generation statistics dict.
"""

from __future__ import annotations

import json
import pathlib
import random
import subprocess
from typing import Any

import h5py
import numpy as np
from omegaconf import OmegaConf

from policy_doctor.curation_pipeline.base_step import PipelineStep
from policy_doctor.mimicgen.eef import extract_eef_xyz_from_hdf5
from policy_doctor.mimicgen.materializer import RobomimicSeedMaterializer
from policy_doctor.mimicgen.seed_trajectory import MimicGenSeedTrajectory
from policy_doctor.paths import MIMICGEN_CONDA_ENV_NAME, PROJECT_ROOT

_DEFAULT_SOURCE_DATASET = (
    "data/source/mimicgen/core_datasets/square/demo_src_square_task_D1/demo.hdf5"
)
_GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "run_mimicgen_generate.py"


def _list_demo_keys(hdf5_path: pathlib.Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted(k for k in f["data"].keys() if k.startswith("demo_"))


class GenerateMimicgenDemosStep(PipelineStep[dict]):
    """Generate MimicGen demos from one seed trajectory and extract EEF data.

    Dispatch model:
      1. Load seed trajectory (h5py only, runs in policy_doctor env).
      2. Materialize to ``step_dir/seed_demo.hdf5`` (h5py only).
      3. Run ``conda run -n mimicgen python scripts/run_mimicgen_generate.py``
         which calls ``prepare_src_dataset`` and ``generate_dataset`` in the
         correct MuJoCo / robosuite environment.
      4. Load EEF trajectories from the generated HDF5 (h5py only).
      5. Return JSON-serialisable result dict.
    """

    name = "generate_mimicgen_demos"

    def compute(self) -> dict[str, Any]:
        cfg_mg = OmegaConf.select(self.cfg, "mimicgen_datagen") or {}

        # episode_budget is the canonical name; num_trials is the legacy fallback.
        num_trials: int = int(
            OmegaConf.select(cfg_mg, "episode_budget")
            or OmegaConf.select(cfg_mg, "num_trials")
            or 50
        )
        output_dir_rel: str = (
            OmegaConf.select(cfg_mg, "output_dir") or "data/outputs/mimicgen_datagen"
        )
        task_name: str = OmegaConf.select(cfg_mg, "task_name") or "square"
        env_interface_name: str = OmegaConf.select(cfg_mg, "env_interface_name") or "MG_Square"
        env_interface_type: str = OmegaConf.select(cfg_mg, "env_interface_type") or "robosuite"

        self.step_dir.mkdir(parents=True, exist_ok=True)

        # --- Auto-wire: prefer SelectMimicgenSeedStep, fall back to SelectMimicgenSeedFromGraphStep ---
        from policy_doctor.curation_pipeline.steps.select_mimicgen_seed import (
            SelectMimicgenSeedStep,
        )
        from policy_doctor.curation_pipeline.steps.select_mimicgen_seed_from_graph import (
            SelectMimicgenSeedFromGraphStep,
        )
        new_select_step = SelectMimicgenSeedStep(self.cfg, self.run_dir)
        old_select_step = SelectMimicgenSeedFromGraphStep(self.cfg, self.run_dir)

        if new_select_step.is_done():
            select_result = new_select_step.load()
            seed_hdf5 = pathlib.Path(select_result["seed_hdf5_path"])
            seed_demo_key = "demo_0"
            seed_source = select_result.get("heuristic", "select_mimicgen_seed")
            print(
                f"  [generate_mimicgen_demos] using heuristic-selected seed: "
                f"heuristic={select_result.get('heuristic')}  "
                f"rollout={select_result.get('rollout_idx')}  "
                f"seed_hdf5={seed_hdf5}"
            )
        elif old_select_step.is_done():
            select_result = old_select_step.load()
            seed_hdf5 = pathlib.Path(select_result["seed_hdf5_path"])
            seed_demo_key = "demo_0"
            seed_source = "graph"
            print(
                f"  [generate_mimicgen_demos] using graph-selected seed (legacy step): "
                f"rollout={select_result.get('selected_rollout_idx')}  "
                f"path_prob={select_result.get('selected_path_prob', 0):.3f}  "
                f"seed_hdf5={seed_hdf5}"
            )
        else:
            # --- Fallback: load from source dataset ---
            seed_source = "source_dataset"
            source_dataset_rel = (
                OmegaConf.select(cfg_mg, "source_dataset_path") or _DEFAULT_SOURCE_DATASET
            )
            source_dataset = self.repo_root / source_dataset_rel
            if not source_dataset.exists():
                raise FileNotFoundError(
                    f"[generate_mimicgen_demos] source dataset not found: {source_dataset}\n"
                    "Set mimicgen_datagen.source_dataset_path in your config, or run "
                    "SelectMimicgenSeedFromGraphStep first."
                )

            seed_demo_key_cfg: str = OmegaConf.select(cfg_mg, "seed_demo_key") or "demo_0"
            seed_random_seed: int = int(OmegaConf.select(cfg_mg, "seed_random_seed") or 42)

            available_keys = _list_demo_keys(source_dataset)
            if not available_keys:
                raise RuntimeError(f"No demo_* keys found in {source_dataset}")

            if seed_demo_key_cfg == "random":
                rng = random.Random(seed_random_seed)
                seed_demo_key = rng.choice(available_keys)
                print(
                    f"  [generate_mimicgen_demos] random seed_demo_key={seed_demo_key!r}"
                    f" (seed={seed_random_seed}, choices={len(available_keys)})"
                )
            else:
                if seed_demo_key_cfg not in available_keys:
                    raise ValueError(
                        f"seed_demo_key={seed_demo_key_cfg!r} not found in {source_dataset}. "
                        f"Available: {available_keys[:10]}"
                    )
                seed_demo_key = seed_demo_key_cfg

            print(f"  [generate_mimicgen_demos] loading seed demo: {seed_demo_key}")
            traj = MimicGenSeedTrajectory.from_robomimic_hdf5_demo(
                source_dataset, demo_key=seed_demo_key
            )

            seed_hdf5 = self.step_dir / "seed_demo.hdf5"
            mat = RobomimicSeedMaterializer()
            mat.write_source_dataset(
                states=traj.states,
                actions=traj.actions,
                env_meta=traj.env_meta,
                output_path=seed_hdf5,
                model_file=traj.model_file,
            )
            print(f"  [generate_mimicgen_demos] seed HDF5 written: {seed_hdf5}")

        # --- Output directory for generation ---
        gen_output_dir = self.repo_root / output_dir_rel / seed_demo_key
        gen_output_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            print(f"[dry_run] GenerateMimicgenDemosStep seed={seed_demo_key} "
                  f"num_trials={num_trials} output={gen_output_dir}")
            return {
                "seed_demo_key": seed_demo_key,
                "seed_source": seed_source,
                "dry_run": True,
                "generated_hdf5_path": str(gen_output_dir / "demo.hdf5"),
                "stats": {},
                "seed_eef_xyz": [],
                "generated_eef_xyz": [],
            }

        # --- Dispatch to mimicgen conda env ---
        cmd = [
            "conda", "run", "-n", MIMICGEN_CONDA_ENV_NAME, "--no-capture-output",
            "python", str(_GENERATE_SCRIPT),
            "--seed_hdf5", str(seed_hdf5),
            "--output_dir", str(gen_output_dir),
            "--task_name", task_name,
            "--env_interface_name", env_interface_name,
            "--env_interface_type", env_interface_type,
            "--num_trials", str(num_trials),
        ]
        print(f"  [generate_mimicgen_demos] running: {' '.join(cmd[:8])} ...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"[generate_mimicgen_demos] mimicgen subprocess failed "
                f"(exit={result.returncode}). See output above."
            )

        # --- Read stats ---
        stats_path = gen_output_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
        else:
            stats = {}

        # --- Extract EEF trajectories ---
        # Seed demo EEF (from prepared seed_hdf5 which now has datagen_info)
        seed_eef_xyz = extract_eef_xyz_from_hdf5(seed_hdf5)

        # Generated demos EEF (successes)
        generated_hdf5 = gen_output_dir / "demo.hdf5"
        generated_eef_xyz: list[np.ndarray] = []
        if generated_hdf5.exists():
            generated_eef_xyz = extract_eef_xyz_from_hdf5(generated_hdf5)
        else:
            print(f"  [generate_mimicgen_demos] WARNING: generated demo.hdf5 not found at {generated_hdf5}")

        # Failed demos EEF
        failed_hdf5 = gen_output_dir / "demo_failed.hdf5"
        failed_eef_xyz: list[np.ndarray] = []
        if failed_hdf5.exists():
            failed_eef_xyz = extract_eef_xyz_from_hdf5(failed_hdf5)

        print(
            f"  [generate_mimicgen_demos] done. "
            f"seed_eef_demos={len(seed_eef_xyz)} "
            f"success_eef_demos={len(generated_eef_xyz)} "
            f"failed_eef_demos={len(failed_eef_xyz)} "
            f"stats={stats.get('num_success', '?')}/{stats.get('num_attempts', '?')}"
        )

        return {
            "seed_demo_key": seed_demo_key,
            "seed_source": seed_source,
            "generated_hdf5_path": str(generated_hdf5),
            "failed_hdf5_path": str(failed_hdf5) if failed_hdf5.exists() else None,
            "stats": stats,
            # Serialise as nested lists for JSON
            "seed_eef_xyz": [arr.tolist() for arr in seed_eef_xyz],
            "generated_eef_xyz": [arr.tolist() for arr in generated_eef_xyz],
            "failed_eef_xyz": [arr.tolist() for arr in failed_eef_xyz],
        }
