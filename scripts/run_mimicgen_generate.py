"""Subprocess entry point for MimicGen prepare + generate pipeline.

This script is called by :class:`~policy_doctor.curation_pipeline.steps
.generate_mimicgen_demos.GenerateMimicgenDemosStep` via::

    conda run -n mimicgen python scripts/run_mimicgen_generate.py \\
        --seed_hdf5   /path/to/seed_demo.hdf5 \\
        --output_dir  /path/to/generation_output \\
        --task_name   square \\
        --env_interface_name  MG_Square \\
        --env_interface_type  robosuite \\
        --num_trials  50

It must run under the **mimicgen** conda environment (MuJoCo 2.3.x, pinned
robosuite / robomimic) because ``prepare_src_dataset`` replays trajectories
through the simulator.

Outputs written to ``output_dir``:
* ``demo.hdf5``         — generated demonstrations (merged)
* ``demo_failed.hdf5``  — failed demonstrations (when present)
* ``stats.json``        — generation statistics (success rate, counts, timing)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _ensure_mimicgen_on_path() -> None:
    """Add vendored MimicGen to sys.path so ``import mimicgen`` resolves."""
    this = Path(__file__).resolve()
    for parent in this.parents:
        candidate = parent / "third_party" / "mimicgen"
        if candidate.is_dir():
            s = str(candidate.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return


def _apply_robomimic_base_env_shim() -> None:
    """Patch robomimic EnvRobosuite.base_env if missing (MimicGen compatibility)."""
    try:
        import robomimic.envs.env_robosuite as er
    except ImportError:
        return
    if not hasattr(er.EnvRobosuite, "base_env"):
        er.EnvRobosuite.base_env = property(lambda self: self.env)  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run MimicGen prepare_src_dataset + generate_dataset for one seed demo."
    )
    # --- Identity ---
    parser.add_argument("--seed_hdf5", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", default="square")
    parser.add_argument("--env_interface_name", default="MG_Square")
    parser.add_argument("--env_interface_type", default="robosuite")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--guarantee", action="store_true", default=False)
    parser.add_argument("--filter_key", default=None)

    # --- Trajectory variance knobs ---
    parser.add_argument("--action_noise", type=float, default=0.05,
                        help="Per-step action noise amplitude (0.0 = deterministic)")
    parser.add_argument("--subtask_term_offset_lo", type=int, default=0,
                        help="subtask_term_offset_range lower bound for subtask_1")
    parser.add_argument("--subtask_term_offset_hi", type=int, default=0,
                        help="subtask_term_offset_range upper bound for subtask_1")
    parser.add_argument("--nn_k", type=int, default=1,
                        help="Nearest-neighbour top-k for subtask_1 source selection")
    parser.add_argument("--interpolate_from_last_target_pose", action="store_true", default=False,
                        help="Interpolate from last commanded target pose (more variance)")
    parser.add_argument("--transform_first_robot_pose", action="store_true", default=False)
    parser.add_argument("--num_interpolation_steps", type=int, default=5)
    parser.add_argument("--num_fixed_steps", type=int, default=0)

    # --- Object pose constraints ---
    parser.add_argument("--seed_object_poses", type=str, default=None,
                        help='JSON dict of world-frame seed poses auto-read from the source HDF5. '
                             'Schema: {object_name: {x: float, y: float, z_rot: float}}.')
    parser.add_argument("--object_pose_ranges", type=str, default=None,
                        help='JSON dict of per-axis offset ranges from the seed pose. '
                             'Schema: {object_name: {x: [lo,hi]|null, y: [lo,hi]|null, '
                             'z_rot: [lo,hi]|null}}. null per-axis = leave that axis with '
                             'its original env random range. null overall object entry = '
                             'pin all axes exactly ([0,0] offsets).')

    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    _ensure_mimicgen_on_path()
    _apply_robomimic_base_env_shim()

    import mimicgen  # noqa: F401
    from mimicgen.configs import config_factory
    from mimicgen.scripts.generate_dataset import generate_dataset
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

    seed_hdf5 = Path(args.seed_hdf5).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: prepare_src_dataset (annotates seed in-place) ---
    print(f"[run_mimicgen_generate] prepare_src_dataset: {seed_hdf5}")
    prepare_src_dataset(
        dataset_path=str(seed_hdf5),
        env_interface_name=args.env_interface_name,
        env_interface_type=args.env_interface_type,
        filter_key=args.filter_key,
        n=None,
        output_path=None,
    )
    print("[run_mimicgen_generate] prepare_src_dataset done.")

    # --- Optional: constrain object initial poses relative to seed ---
    if args.seed_object_poses:
        import json as _json
        import h5py as _h5py
        import robosuite.environments as _renv

        # World-frame seed poses: {object_name: {x, y, z_rot}}
        seed_poses: dict[str, dict[str, float]] = _json.loads(args.seed_object_poses)
        # Per-axis offset ranges: {object_name: {x: [lo,hi]|null, y: [lo,hi]|null, z_rot: [lo,hi]|null}}
        # None overall = pin exactly (all [0,0])
        pose_ranges: dict[str, dict[str, list | None]] = (
            _json.loads(args.object_pose_ranges) if args.object_pose_ranges else {}
        )

        # Resolve env class from seed HDF5 env_meta (task-agnostic)
        with _h5py.File(str(seed_hdf5), "r") as _f:
            _env_meta = _json.loads(_f["data"].attrs["env_args"])
        env_name = _env_meta.get("env_name", "")
        env_cls = _renv.REGISTERED_ENVS.get(env_name)

        if env_cls is None:
            print(f"[run_mimicgen_generate] WARNING: env {env_name!r} not in registry; "
                  "skipping pose constraint")
        elif not hasattr(env_cls, "_get_initial_placement_bounds"):
            print(f"[run_mimicgen_generate] WARNING: {env_name} has no "
                  "_get_initial_placement_bounds; skipping pose constraint")
        else:
            _orig_bounds = env_cls._get_initial_placement_bounds

            def _constrained_bounds(
                self,
                _seed=seed_poses,
                _ranges=pose_ranges,
                _orig=_orig_bounds,
            ):
                bounds = _orig(self)
                for bounds_key in bounds:
                    # Match bounds_key to a seed object: exact match or substring
                    matched_seed = next(
                        (seed for obj_name, seed in _seed.items()
                         if bounds_key == obj_name
                         or bounds_key in obj_name
                         or obj_name in bounds_key),
                        None,
                    )
                    if matched_seed is None:
                        continue
                    # Per-object axis ranges (default: pin exactly = [0, 0] offset).
                    # Fuzzy-match ranges keys to the bounds/seed key too, since config
                    # keys (e.g. "nut") may differ from HDF5 object names ("square_nut").
                    obj_ranges = next(
                        (comps for range_key, comps in _ranges.items()
                         if bounds_key == range_key
                         or bounds_key in range_key
                         or range_key in bounds_key),
                        {},
                    )

                    for axis in ("x", "y", "z_rot"):
                        if axis not in bounds[bounds_key]:
                            continue
                        axis_range = obj_ranges.get(axis, [0.0, 0.0])
                        if axis_range is None:
                            # null = leave env's original random range for this axis
                            continue
                        lo_offset, hi_offset = axis_range[0], axis_range[1]
                        if axis in ("x", "y"):
                            # bounds x/y are relative to the env reference point;
                            # seed value is world-frame, so subtract reference to get relative
                            ref_idx = 0 if axis == "x" else 1
                            ref_val = float(bounds[bounds_key]["reference"][ref_idx])
                            seed_rel = matched_seed[axis] - ref_val
                        else:
                            seed_rel = matched_seed[axis]
                        bounds[bounds_key][axis] = (seed_rel + lo_offset, seed_rel + hi_offset)
                return bounds

            env_cls._get_initial_placement_bounds = _constrained_bounds

            # Log a compact summary of what will be constrained
            parts = []
            for obj, seed in seed_poses.items():
                obj_ranges = next(
                    (comps for rk, comps in pose_ranges.items()
                     if obj == rk or obj in rk or rk in obj),
                    {},
                )
                axis_parts = []
                for axis in ("x", "y", "z_rot"):
                    r = obj_ranges.get(axis, [0.0, 0.0])
                    if r is None:
                        axis_parts.append(f"{axis}=free")
                    elif r == [0.0, 0.0]:
                        axis_parts.append(f"{axis}={seed.get(axis, 0):.3f}(pinned)")
                    else:
                        sv = seed.get(axis, 0)
                        axis_parts.append(f"{axis}=[{sv+r[0]:.3f},{sv+r[1]:.3f}]")
                parts.append(f"{obj}({', '.join(axis_parts)})")
            print(f"[run_mimicgen_generate] constrained object poses on {env_name}: "
                  + "; ".join(parts))

    # --- Step 2: build MimicGen config ---
    gen_tmp = output_dir / "_gen_tmp"

    cfg = config_factory(args.task_name, "robosuite")
    cfg.experiment.name = "generated"
    cfg.experiment.source.dataset_path = str(seed_hdf5)
    cfg.experiment.source.n = None
    cfg.experiment.generation.path = str(gen_tmp)
    cfg.experiment.generation.num_trials = args.num_trials
    cfg.experiment.generation.guarantee = args.guarantee
    cfg.experiment.generation.keep_failed = True
    cfg.experiment.generation.interpolate_from_last_target_pose = args.interpolate_from_last_target_pose
    cfg.experiment.generation.transform_first_robot_pose = args.transform_first_robot_pose
    cfg.experiment.generation.select_src_per_subtask = False
    cfg.experiment.render_video = False
    cfg.experiment.num_demo_to_render = 0
    cfg.experiment.num_fail_demo_to_render = 0
    cfg.experiment.max_num_failures = max(args.num_trials * 2, 100)
    cfg.experiment.log_every_n_attempts = 100
    cfg.experiment.task.name = None
    cfg.obs.collect_obs = True
    cfg.obs.camera_names = []

    cfg.task.task_spec.subtask_1 = dict(
        object_ref="square_nut",
        subtask_term_signal="grasp",
        subtask_term_offset_range=(args.subtask_term_offset_lo, args.subtask_term_offset_hi),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs=dict(nn_k=args.nn_k),
        action_noise=args.action_noise,
        num_interpolation_steps=args.num_interpolation_steps,
        num_fixed_steps=args.num_fixed_steps,
        apply_noise_during_interpolation=False,
    )
    cfg.task.task_spec.subtask_2 = dict(
        object_ref="square_peg",
        subtask_term_signal=None,
        subtask_term_offset_range=None,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=args.action_noise,
        num_interpolation_steps=args.num_interpolation_steps,
        num_fixed_steps=args.num_fixed_steps,
        apply_noise_during_interpolation=False,
    )

    print(f"[run_mimicgen_generate] generate_dataset: num_trials={args.num_trials}")
    stats = generate_dataset(cfg, auto_remove_exp=True, render=False, video_path=None)

    # --- Step 3: copy outputs and write stats ---
    for fname in ("demo.hdf5", "demo_failed.hdf5"):
        src = gen_tmp / "generated" / fname
        if src.is_file():
            shutil.copy2(src, output_dir / fname)
            print(f"[run_mimicgen_generate] {fname} → {output_dir / fname}")
        elif fname == "demo.hdf5":
            print(f"[run_mimicgen_generate] WARNING: {fname} not found at {src}")

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats or {}, f, indent=2, default=str)
    print(f"[run_mimicgen_generate] stats → {stats_path}")

    if gen_tmp.exists():
        shutil.rmtree(gen_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
