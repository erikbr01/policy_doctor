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

    Trajectory variance knobs (see docs/mimicgen_data_generation_parameters.md):
    action_noise                     Per-step action noise (default 0.05).
    subtask_term_offset_range        [lo, hi] random boundary offset for subtask_1
                                     (default [0, 0]).
    nn_k                             Nearest-neighbour top-k for subtask_1 source
                                     selection (default 1).
    interpolate_from_last_target_pose  (default false)
    transform_first_robot_pose         (default false)
    num_interpolation_steps            (default 5)
    num_fixed_steps                    (default 0)

    Object pose constraints:
    fix_initial_object_poses  If true, constrain each object's initial pose relative
                              to the seed demo on every env.reset() (default false).
    object_pose_ranges        Per-object, per-axis offset ranges from the seed pose.
                              Schema: {object_name: {x: [lo,hi]|null, y: [lo,hi]|null,
                              z_rot: [lo,hi]|null}}.
                              [0, 0] = pin exactly to seed value.
                              [-d, d] = uniform ±d around seed value.
                              null per-axis = leave that axis with env's random range.
                              null overall = pin all axes exactly to seed ([0,0]).

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


def _read_all_object_poses_from_hdf5(
    hdf5_path: pathlib.Path,
) -> dict[str, dict[str, float]]:
    """Return world-frame pose at t=0 for every object in datagen_info/object_poses.

    Works for any MimicGen task — reads whatever objects are present in the HDF5.
    Returns world-frame (x, y, z_rot) so that the subprocess can subtract the
    env-specific reference to compute relative bounds offsets.

    Args:
        hdf5_path: Prepared source or generated HDF5 (must have datagen_info).

    Returns:
        Dict mapping object_name → {x: float, y: float, z_rot: float} in world frame.

    Raises:
        RuntimeError: if datagen_info/object_poses is missing (file not prepared).
    """
    result: dict[str, dict[str, float]] = {}
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
        if not demo_keys:
            raise RuntimeError(f"No demo_* keys found in {hdf5_path}")
        poses_grp_key = f"data/{demo_keys[0]}/datagen_info/object_poses"
        if poses_grp_key not in f:
            raise RuntimeError(
                f"datagen_info/object_poses not found in {hdf5_path} — "
                "has prepare_src_dataset been run on this file?"
            )
        for obj_name in f[poses_grp_key].keys():
            poses = np.array(f[f"{poses_grp_key}/{obj_name}"])  # (T, 4, 4)
            pos = poses[0, :3, 3]          # world-frame translation
            R = poses[0, :3, :3]           # rotation matrix
            result[obj_name] = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z_rot": float(np.arctan2(R[1, 0], R[0, 0])),
            }
    return result


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

        # --- Basic generation params ---
        num_trials: int = int(OmegaConf.select(cfg_mg, "num_trials") or 50)
        output_dir_rel: str = (
            OmegaConf.select(cfg_mg, "output_dir") or "data/outputs/mimicgen_datagen"
        )
        task_name: str = OmegaConf.select(cfg_mg, "task_name") or "square"
        env_interface_name: str = OmegaConf.select(cfg_mg, "env_interface_name") or "MG_Square"
        env_interface_type: str = OmegaConf.select(cfg_mg, "env_interface_type") or "robosuite"

        # --- Variance knobs ---
        # Use OmegaConf.select with default= to avoid `0.0 or fallback` falsiness bugs.
        action_noise: float = float(
            OmegaConf.select(cfg_mg, "action_noise", default=0.05)
        )
        offset_range = OmegaConf.select(cfg_mg, "subtask_term_offset_range", default=[0, 0])
        offset_lo: int = int(offset_range[0])
        offset_hi: int = int(offset_range[1])
        nn_k: int = int(OmegaConf.select(cfg_mg, "nn_k", default=1))
        interp_from_last: bool = bool(
            OmegaConf.select(cfg_mg, "interpolate_from_last_target_pose", default=False)
        )
        transform_first: bool = bool(
            OmegaConf.select(cfg_mg, "transform_first_robot_pose", default=False)
        )
        num_interp_steps: int = int(OmegaConf.select(cfg_mg, "num_interpolation_steps", default=5))
        num_fixed_steps: int = int(OmegaConf.select(cfg_mg, "num_fixed_steps", default=0))

        # --- Object pose constraints ---
        fix_initial_object_poses: bool = bool(
            OmegaConf.select(cfg_mg, "fix_initial_object_poses") or False
        )
        object_pose_ranges = OmegaConf.select(cfg_mg, "object_pose_ranges")  # nested dict or None

        self.step_dir.mkdir(parents=True, exist_ok=True)

        # --- Auto-wire: check if SelectMimicgenSeedFromGraphStep has been run ---
        from policy_doctor.curation_pipeline.steps.select_mimicgen_seed_from_graph import (
            SelectMimicgenSeedFromGraphStep,
        )
        select_step = SelectMimicgenSeedFromGraphStep(self.cfg, self.run_dir)
        if select_step.is_done():
            select_result = select_step.load()
            seed_hdf5 = pathlib.Path(select_result["seed_hdf5_path"])
            seed_demo_key = "demo_0"
            seed_source = "graph"
            print(
                f"  [generate_mimicgen_demos] using graph-selected seed: "
                f"rollout={select_result.get('selected_rollout_idx')}  "
                f"path_prob={select_result.get('selected_path_prob', 0):.3f}  "
                f"seed_hdf5={seed_hdf5}"
            )
        else:
            # --- Fallback: load from source dataset ---
            seed_source = "source_dataset"
            source_dataset_str = (
                OmegaConf.select(cfg_mg, "source_dataset_path") or _DEFAULT_SOURCE_DATASET
            )
            source_dataset_path = pathlib.Path(source_dataset_str)
            # If relative, try PROJECT_ROOT first (MimicGen source data lives there),
            # then fall back to REPO_ROOT (third_party/cupid).
            if not source_dataset_path.is_absolute():
                candidate_project = PROJECT_ROOT / source_dataset_path
                candidate_repo = self.repo_root / source_dataset_path
                if candidate_project.exists():
                    source_dataset = candidate_project
                elif candidate_repo.exists():
                    source_dataset = candidate_repo
                else:
                    raise FileNotFoundError(
                        f"[generate_mimicgen_demos] source dataset not found at:\n"
                        f"  {candidate_project}\n"
                        f"  {candidate_repo}\n"
                        "Set mimicgen_datagen.source_dataset_path in your config, or run "
                        "SelectMimicgenSeedFromGraphStep first."
                    )
            else:
                source_dataset = source_dataset_path
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

        # --- Auto-read seed world poses for all objects (always when pose-fixing is on) ---
        seed_object_poses: dict | None = None
        if fix_initial_object_poses:
            # seed_hdf5 may not yet have datagen_info (prepare_src_dataset runs in the
            # subprocess), so fall back to reading from the source dataset directly.
            read_target = seed_hdf5
            try:
                seed_object_poses = _read_all_object_poses_from_hdf5(read_target)
            except RuntimeError:
                # seed_hdf5 not yet prepared; fall back to source dataset
                src_str = OmegaConf.select(cfg_mg, "source_dataset_path") or _DEFAULT_SOURCE_DATASET
                src_p = pathlib.Path(src_str)
                if not src_p.is_absolute():
                    src_p = PROJECT_ROOT / src_p if (PROJECT_ROOT / src_p).exists() else self.repo_root / src_p
                read_target = src_p
                seed_object_poses = _read_all_object_poses_from_hdf5(read_target)
            summary = ", ".join(
                f"{obj}=({', '.join(f'{k}:{v:.3f}' for k, v in comps.items())})"
                for obj, comps in seed_object_poses.items()
            )
            print(
                f"  [generate_mimicgen_demos] seed object poses from "
                f"{read_target.name}: {summary}"
            )
            if object_pose_ranges is None:
                print("  [generate_mimicgen_demos] object_pose_ranges=null → "
                      "all axes pinned exactly to seed ([0,0] offsets)")

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
                "failed_eef_xyz": [],
            }

        # --- Dispatch to mimicgen conda env ---
        cmd = [
            "conda", "run", "-n", MIMICGEN_CONDA_ENV_NAME, "--no-capture-output",
            "python", str(_GENERATE_SCRIPT),
            "--seed_hdf5",          str(seed_hdf5),
            "--output_dir",         str(gen_output_dir),
            "--task_name",          task_name,
            "--env_interface_name", env_interface_name,
            "--env_interface_type", env_interface_type,
            "--num_trials",         str(num_trials),
            "--action_noise",       str(action_noise),
            "--subtask_term_offset_lo", str(offset_lo),
            "--subtask_term_offset_hi", str(offset_hi),
            "--nn_k",               str(nn_k),
            "--num_interpolation_steps", str(num_interp_steps),
            "--num_fixed_steps",    str(num_fixed_steps),
        ]
        if interp_from_last:
            cmd.append("--interpolate_from_last_target_pose")
        if transform_first:
            cmd.append("--transform_first_robot_pose")
        if fix_initial_object_poses and seed_object_poses:
            import json as _json
            cmd += ["--seed_object_poses", _json.dumps(seed_object_poses)]
            if object_pose_ranges is not None:
                ranges_plain = OmegaConf.to_container(object_pose_ranges, resolve=True)
                cmd += ["--object_pose_ranges", _json.dumps(ranges_plain)]
            # If object_pose_ranges is None, the subprocess defaults to [0,0] for all axes

        print(f"  [generate_mimicgen_demos] running generation ...")
        print(f"    action_noise={action_noise}  offset=({offset_lo},{offset_hi})  nn_k={nn_k}")
        print(f"    interp_from_last={interp_from_last}  transform_first={transform_first}")
        print(f"    num_interp={num_interp_steps}  num_fixed={num_fixed_steps}")
        print(f"    fix_initial_object_poses={fix_initial_object_poses}  "
              f"object_pose_ranges={object_pose_ranges}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"[generate_mimicgen_demos] mimicgen subprocess failed "
                f"(exit={result.returncode}). See output above."
            )

        # --- Read stats ---
        stats_path = gen_output_dir / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

        # --- Extract EEF trajectories ---
        seed_eef_xyz = extract_eef_xyz_from_hdf5(seed_hdf5)

        generated_hdf5 = gen_output_dir / "demo.hdf5"
        generated_eef_xyz: list[np.ndarray] = []
        if generated_hdf5.exists():
            generated_eef_xyz = extract_eef_xyz_from_hdf5(generated_hdf5)
        else:
            print(f"  [generate_mimicgen_demos] WARNING: generated demo.hdf5 not found")

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
            "seed_eef_xyz":      [arr.tolist() for arr in seed_eef_xyz],
            "generated_eef_xyz": [arr.tolist() for arr in generated_eef_xyz],
            "failed_eef_xyz":    [arr.tolist() for arr in failed_eef_xyz],
        }
