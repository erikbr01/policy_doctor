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
    source_dataset_path      Path to the robomimic HDF5 with source demos.
                             Defaults to the standard MimicGen Square D1 path.
                             Ignored when SelectMimicgenSeedFromGraphStep is done.
    use_full_source_dataset  If true, pass the entire source HDF5 to MimicGen
                             without materializing a single demo. Required for
                             D1-style tasks where nn_k > 1 needs a diverse source
                             pool (default false).
    seed_demo_key            Demo to use as seed. Default ``"demo_0"``.
                             Set to ``"random"`` to draw randomly (deterministic
                             via ``seed_random_seed``).
                             Ignored when use_full_source_dataset=true or
                             SelectMimicgenSeedFromGraphStep is done.
    seed_random_seed         RNG seed used when ``seed_demo_key="random"`` (default 42).
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
from policy_doctor.paths import MIMICGEN_CONDA_ENV_NAME as _MIMICGEN_CONDA_ENV_DEFAULT, PROJECT_ROOT

_DEFAULT_SOURCE_DATASET = (
    "data/source/mimicgen/core_datasets/square/demo_src_square_task_D1/demo.hdf5"
)
_GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "run_mimicgen_generate.py"


def _list_demo_keys(hdf5_path: pathlib.Path) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        return sorted(k for k in f["data"].keys() if k.startswith("demo_"))


def _extract_single_seed(
    src_hdf5: pathlib.Path,
    demo_key: str,
    dst_hdf5: pathlib.Path,
) -> None:
    """Copy one demo from a multi-demo seed HDF5 into a new single-demo file as demo_0.

    The destination file always contains exactly one demo keyed ``demo_0``.
    Top-level attributes from ``data`` (e.g. ``env_args``) are preserved.
    """
    dst_hdf5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_hdf5, "r") as src_f:
        if demo_key not in src_f["data"]:
            raise KeyError(f"{demo_key!r} not found in {src_hdf5}")
        with h5py.File(dst_hdf5, "w") as dst_f:
            data_grp = dst_f.require_group("data")
            # Copy top-level data attributes (env_args etc.)
            for attr_key, attr_val in src_f["data"].attrs.items():
                data_grp.attrs[attr_key] = attr_val
            # Copy any non-demo groups at the top level (e.g. mask)
            for name, item in src_f.items():
                if name == "data":
                    continue
                if isinstance(item, h5py.Group):
                    _copy_group_standalone(item, dst_f.require_group(name))
            # Copy the requested demo as demo_0
            demo_dst = data_grp.require_group("demo_0")
            _copy_group_standalone(src_f[f"data/{demo_key}"], demo_dst)
            data_grp.attrs["total"] = 1


def _copy_group_standalone(src: "h5py.Group", dst: "h5py.Group") -> None:
    """Recursively copy an h5py Group, reading datasets as numpy arrays."""
    for attr_key, attr_val in src.attrs.items():
        dst.attrs[attr_key] = attr_val
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            _copy_group_standalone(item, dst.require_group(name))
        else:
            data = item[()]
            kwargs: dict = {}
            if item.chunks is not None:
                kwargs["chunks"] = item.chunks
            if item.compression is not None:
                kwargs["compression"] = item.compression
                if item.compression_opts is not None:
                    kwargs["compression_opts"] = item.compression_opts
            ds = dst.create_dataset(name, data=data, **kwargs)
            for attr_key, attr_val in item.attrs.items():
                ds.attrs[attr_key] = attr_val


def _merge_hdf5s(hdf5_paths: list[pathlib.Path], output_path: pathlib.Path) -> int:
    """Merge multiple generated demo HDF5 files into one, renaming demos to avoid collisions.

    Demos from each file are appended in order as ``demo_0``, ``demo_1``, …  Top-level
    ``data`` attributes are taken from the first non-empty file.  Returns total demo count.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [p for p in hdf5_paths if p.exists()]
    if not existing:
        raise RuntimeError(
            f"[_merge_hdf5s] All {len(hdf5_paths)} per-seed generation jobs produced no output. "
            f"Expected files: {[str(p) for p in hdf5_paths]}. "
            f"Check subprocess logs above for failure details."
        )

    counter = 0
    with h5py.File(output_path, "w") as out_f:
        data_grp = out_f.require_group("data")
        attrs_set = False
        for src_path in existing:
            with h5py.File(src_path, "r") as src_f:
                if "data" not in src_f:
                    continue
                if not attrs_set:
                    for attr_key, attr_val in src_f["data"].attrs.items():
                        data_grp.attrs[attr_key] = attr_val
                    attrs_set = True
                demo_keys = sorted(k for k in src_f["data"].keys() if k.startswith("demo_"))
                for demo_key in demo_keys:
                    new_key = f"demo_{counter}"
                    demo_dst = data_grp.require_group(new_key)
                    _copy_group_standalone(src_f[f"data/{demo_key}"], demo_dst)
                    counter += 1
        data_grp.attrs["total"] = counter
    return counter


def _subsample_hdf5(path: pathlib.Path, n: int, rng_seed: int = 0) -> int:
    """Randomly subsample an HDF5 file in-place to at most *n* demos.

    If the file already has <= n demos, it is left unchanged.  Returns the
    final demo count.  Uses a fixed rng_seed so the subsample is reproducible.
    """
    import random as _random
    with h5py.File(path, "r") as f:
        all_keys = sorted(k for k in f["data"].keys() if k.startswith("demo_"))

    if len(all_keys) <= n:
        return len(all_keys)

    rng = _random.Random(rng_seed)
    keep_keys = set(rng.sample(all_keys, n))
    remove_keys = [k for k in all_keys if k not in keep_keys]

    with h5py.File(path, "a") as f:
        for k in remove_keys:
            del f[f"data/{k}"]
        # Rename remaining keys to dense demo_0..demo_{n-1}.
        # Use a temporary prefix to avoid collisions when e.g. demo_3 → demo_0
        # would collide with an existing demo_0.
        remaining = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
        for old_key in remaining:
            f["data"].move(old_key, f"_tmp_{old_key}")
        tmp_keys = sorted(k for k in f["data"].keys() if k.startswith("_tmp_demo_"))
        for i, tmp_key in enumerate(tmp_keys):
            f["data"].move(tmp_key, f"demo_{i}")
        f["data"].attrs["total"] = n

    return n


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
        # success_budget: run enough trials to expect this many successes.
        # Uses a conservative 40% success rate floor so we over-shoot rather than
        # under-shoot.  episode_budget / num_trials are the legacy trial-count knobs.
        success_budget: int | None = OmegaConf.select(cfg_mg, "success_budget")
        if success_budget is not None:
            success_budget = int(success_budget)
            import math as _math
            # Assume at least 40% success rate; round up and add 20% margin
            num_trials = int(_math.ceil(success_budget / 0.40) * 1.20)
            print(f"  [generate_mimicgen_demos] success_budget={success_budget} → num_trials={num_trials}")
        else:
            num_trials = int(
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

        use_full_source: bool = bool(
            OmegaConf.select(cfg_mg, "use_full_source_dataset", default=False)
        )

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
            # --- Resolve source dataset path ---
            seed_source = "source_dataset"
            source_dataset_str = (
                OmegaConf.select(cfg_mg, "source_dataset_path") or _DEFAULT_SOURCE_DATASET
            )
            source_dataset_path = pathlib.Path(source_dataset_str)
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

            if use_full_source:
                # Pass the full source dataset directly — no materialization.
                # Required for tasks like D1 where nn_k > 1 needs a diverse pool.
                seed_hdf5 = source_dataset
                seed_demo_key = "full_source"
                n_demos = len(_list_demo_keys(source_dataset))
                print(
                    f"  [generate_mimicgen_demos] using full source dataset: "
                    f"{source_dataset.name} ({n_demos} demos)"
                )
            else:
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
        # Use step_dir/output/ so each arm (random, behavior_graph) has its own unique
        # directory and concurrent runs never clobber each other's generated HDF5 files.
        # The config output_dir_rel is intentionally ignored here to avoid collisions.
        gen_output_dir = self.step_dir / "output"
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
        # Allow config override so different worktrees can use different envs.
        mimicgen_env = (
            OmegaConf.select(self.cfg, "data_source.conda_env_datagen")
            or _MIMICGEN_CONDA_ENV_DEFAULT
        )

        def _build_cmd(s_hdf5: pathlib.Path, out_dir: pathlib.Path, n_trials: int) -> list:
            cmd = [
                "conda", "run", "-n", mimicgen_env, "--no-capture-output",
                "python", str(_GENERATE_SCRIPT),
                "--seed_hdf5",          str(s_hdf5),
                "--output_dir",         str(out_dir),
                "--task_name",          task_name,
                "--env_interface_name", env_interface_name,
                "--env_interface_type", env_interface_type,
                "--num_trials",         str(n_trials),
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
                cmd += ["--seed_object_poses", json.dumps(seed_object_poses)]
                if object_pose_ranges is not None:
                    ranges_plain = OmegaConf.to_container(object_pose_ranges, resolve=True)
                    cmd += ["--object_pose_ranges", json.dumps(ranges_plain)]
            return cmd

        print(f"    action_noise={action_noise}  offset=({offset_lo},{offset_hi})  nn_k={nn_k}")
        print(f"    interp_from_last={interp_from_last}  transform_first={transform_first}")
        print(f"    num_interp={num_interp_steps}  num_fixed={num_fixed_steps}")
        print(f"    fix_initial_object_poses={fix_initial_object_poses}  "
              f"object_pose_ranges={object_pose_ranges}")

        # --- Count seeds in seed HDF5 ---
        import h5py as _h5py
        import math as _math
        with _h5py.File(seed_hdf5, "r") as _f:
            n_seeds_in_hdf5 = sum(1 for k in _f["data"].keys() if k.startswith("demo_"))

        # Per-seed mode: when success_budget is set and there are multiple seeds,
        # run a separate generation job for each seed targeting an equal share of
        # successes, then merge all outputs.  This guarantees equal representation
        # from every seed rather than relying on uniform random sampling.
        use_per_seed = success_budget is not None and n_seeds_in_hdf5 > 1

        generated_hdf5 = gen_output_dir / "demo.hdf5"
        failed_hdf5 = gen_output_dir / "demo_failed.hdf5"

        if use_per_seed:
            max_total_trials = success_budget * 20  # hard upper limit
            print(
                f"  [generate_mimicgen_demos] per-seed mode: {n_seeds_in_hdf5} seeds, "
                f"target={success_budget} successes, hard limit={max_total_trials} trials"
            )

            # Pre-extract each seed into its own HDF5 (done once, reused across passes).
            seed_hdf5s: list[tuple[int, pathlib.Path, pathlib.Path]] = []
            for seed_i in range(n_seeds_in_hdf5):
                seed_i_dir = gen_output_dir / f"seed_{seed_i}"
                seed_i_dir.mkdir(parents=True, exist_ok=True)
                seed_i_hdf5 = seed_i_dir / "seed.hdf5"
                _extract_single_seed(seed_hdf5, f"demo_{seed_i}", seed_i_hdf5)
                seed_hdf5s.append((seed_i, seed_i_dir, seed_i_hdf5))

            # Per-seed accumulated stats (one entry per seed, summed across passes).
            per_seed_acc: list[dict] = [{} for _ in range(n_seeds_in_hdf5)]
            per_seed_hdf5s: list[pathlib.Path] = []
            total_successes = 0
            total_trials = 0
            pass_num = 0
            failed_seed_passes: list[tuple[int, int]] = []  # (seed_i, pass_num) pairs

            while total_successes < success_budget and total_trials < max_total_trials:
                pass_num += 1
                remaining_needed = success_budget - total_successes
                remaining_budget = max_total_trials - total_trials

                # Estimate trials per seed using observed rate (40% floor to avoid explosion).
                observed_rate = total_successes / total_trials if total_trials > 0 else 0.40
                effective_rate = max(observed_rate, 0.05)
                trials_per_seed = int(
                    _math.ceil(remaining_needed / n_seeds_in_hdf5 / effective_rate * 1.2)
                )
                # Cap so we never overshoot the total trial budget.
                trials_per_seed = min(
                    trials_per_seed,
                    max(1, remaining_budget // n_seeds_in_hdf5),
                )

                print(
                    f"  [generate_mimicgen_demos] pass {pass_num}: "
                    f"{total_successes}/{success_budget} successes, "
                    f"{total_trials}/{max_total_trials} trials used — "
                    f"running {trials_per_seed} trials/seed"
                )

                for seed_i, seed_i_dir, seed_i_hdf5 in seed_hdf5s:
                    if total_successes >= success_budget or total_trials >= max_total_trials:
                        break

                    # Each pass writes to its own subdirectory to avoid overwriting.
                    pass_out_dir = seed_i_dir / f"_gen_tmp_pass{pass_num:04d}"
                    pass_out_dir.mkdir(parents=True, exist_ok=True)

                    print(
                        f"  [generate_mimicgen_demos] "
                        f"seed {seed_i + 1}/{n_seeds_in_hdf5} pass {pass_num} ..."
                    )
                    res = subprocess.run(
                        _build_cmd(seed_i_hdf5, pass_out_dir, trials_per_seed)
                    )
                    if res.returncode != 0:
                        print(
                            f"  [generate_mimicgen_demos] ERROR: seed {seed_i} "
                            f"pass {pass_num} subprocess failed "
                            f"(exit={res.returncode}), skipping this seed/pass."
                        )
                        failed_seed_passes.append((seed_i, pass_num))
                        total_trials += trials_per_seed  # count as used
                        continue

                    s_path = pass_out_dir / "stats.json"
                    s = json.loads(s_path.read_text()) if s_path.exists() else {}
                    n_succ = s.get("num_success", 0)
                    n_att = s.get("num_attempts", 0)
                    total_successes += n_succ
                    total_trials += n_att

                    # Accumulate per-seed stats across passes.
                    acc = per_seed_acc[seed_i]
                    for k in ("num_success", "num_failures", "num_attempts", "num_problematic"):
                        acc[k] = acc.get(k, 0) + s.get(k, 0)
                    acc["success_rate"] = (
                        100.0 * acc["num_success"] / acc["num_attempts"]
                        if acc["num_attempts"] > 0 else 0.0
                    )
                    acc["failure_rate"] = (
                        100.0 * acc["num_failures"] / acc["num_attempts"]
                        if acc["num_attempts"] > 0 else 0.0
                    )
                    acc.setdefault("generation_path", str(pass_out_dir))
                    for ep_k in ("ep_length_mean", "ep_length_std", "ep_length_max", "ep_length_3std"):
                        if ep_k in s:
                            acc[ep_k] = s[ep_k]  # last pass wins for episode lengths
                    acc["time spent (hrs)"] = s.get("time spent (hrs)", "0.00")

                    if (pass_out_dir / "demo.hdf5").exists():
                        per_seed_hdf5s.append(pass_out_dir / "demo.hdf5")

            if total_successes < success_budget:
                print(
                    f"  [generate_mimicgen_demos] WARNING: hit trial limit "
                    f"({total_trials}/{max_total_trials}) with only "
                    f"{total_successes}/{success_budget} successes."
                )

            if failed_seed_passes:
                n_failed = len(failed_seed_passes)
                n_total_passes = pass_num * n_seeds_in_hdf5
                print(
                    f"  [generate_mimicgen_demos] WARNING: {n_failed}/{n_total_passes} "
                    f"seed/pass jobs failed: {failed_seed_passes}"
                )

            # Merge all per-seed/per-pass outputs, then subsample to success_budget
            # so every arm trains on exactly the same number of generated demos.
            _merge_hdf5s(per_seed_hdf5s, generated_hdf5)
            merged_count = sum(1 for _ in h5py.File(generated_hdf5, "r")["data"].keys()
                               if _.startswith("demo_"))
            if merged_count > success_budget:
                final_count = _subsample_hdf5(generated_hdf5, success_budget)
                print(
                    f"  [generate_mimicgen_demos] subsampled {merged_count} → "
                    f"{final_count} demos (success_budget={success_budget})"
                )

            per_seed_stats = per_seed_acc  # keep variable name for stats block below
            stats = {
                "num_success":  total_successes,
                "num_failures": total_trials - total_successes,
                "num_attempts": total_trials,
                "per_seed_stats": per_seed_stats,
            }
            if stats["num_attempts"] > 0:
                stats["success_rate"] = round(
                    100 * stats["num_success"] / stats["num_attempts"], 1
                )
            (gen_output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

        else:
            # Single-run path (no success_budget, or only one seed)
            print(f"  [generate_mimicgen_demos] running generation (single run, {num_trials} trials) ...")
            result = subprocess.run(_build_cmd(seed_hdf5, gen_output_dir, num_trials))
            if result.returncode != 0:
                raise RuntimeError(
                    f"[generate_mimicgen_demos] mimicgen subprocess failed "
                    f"(exit={result.returncode}). See output above."
                )
            stats_path = gen_output_dir / "stats.json"
            stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

        # --- Extract EEF trajectories ---
        seed_eef_xyz = extract_eef_xyz_from_hdf5(seed_hdf5)

        generated_eef_xyz: list[np.ndarray] = []
        if generated_hdf5.exists():
            generated_eef_xyz = extract_eef_xyz_from_hdf5(generated_hdf5)
        else:
            print(f"  [generate_mimicgen_demos] WARNING: generated demo.hdf5 not found")

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
