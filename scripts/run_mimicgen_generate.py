"""Subprocess entry point for MimicGen prepare + generate pipeline.

This script is called by :class:`~policy_doctor.curation_pipeline.steps
.generate_mimicgen_demos.GenerateMimicgenDemosStep` via::

    conda run -n mimicgen_torch2 python scripts/run_mimicgen_generate.py \\
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
    """Patch robomimic for MimicGen compatibility.

    1. EnvRobosuite.base_env: property missing in robomimic 0.2.x — add it.
    2. EnvUtils.create_env_for_data_processing: robomimic 0.3.0 lacks the
       ``env_class`` kwarg that MimicGen (newer) passes — wrap it to absorb the
       unknown keyword silently (env_class=None is the only value MimicGen passes,
       so dropping it is safe).
    """
    try:
        import robomimic.envs.env_robosuite as er
    except ImportError:
        return
    if not hasattr(er.EnvRobosuite, "base_env"):
        er.EnvRobosuite.base_env = property(lambda self: self.env)  # type: ignore[attr-defined]

    try:
        import inspect
        import robomimic.utils.env_utils as eu
        _orig_create = eu.create_env_for_data_processing
        _accepted = set(inspect.signature(_orig_create).parameters)
        if not {"env_class", "render", "render_offscreen", "use_image_obs", "use_depth_obs"}.issubset(_accepted):
            # robomimic 0.3.0 has a narrower signature — strip unknown kwargs
            def _patched_create(*args, **kwargs):
                filtered = {k: v for k, v in kwargs.items() if k in _accepted}
                return _orig_create(*args, **filtered)
            eu.create_env_for_data_processing = _patched_create
    except Exception:
        pass


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
    parser.add_argument("--subtask_constraints", type=str, default=None,
                        help='JSON dict of per-subtask object-pose constraints. '
                             'Schema: {"<subtask_idx>": {object_name: {x:[lo,hi], y:[lo,hi], '
                             'z_rot:[lo,hi]}}}. After subtask <subtask_idx> executes, any '
                             'trial whose object pose falls outside the constraint is aborted '
                             '(rejected). Uses same world→relative coordinate conversion as '
                             '--object_pose_ranges. Disabled when null (default).')

    # --- Phase-2 chained-warp constraint ---------------------------------
    # When set, swaps in policy_doctor's ChainedWarpDataGenerator: the trial
    # is *early-aborted* the moment subtask <subtask_idx> ends outside the
    # slack box. Later subtasks then warp naturally around the achieved
    # (within-slack) end pose — that's the chained-warp behaviour.
    parser.add_argument("--chained_warp_constraint", type=str, default=None,
                        help='JSON dict for the new constraint-aware generator. '
                             'Schema: {"subtask_idx": int, '
                             '         "target_pose": {obj: {x, y, z_rot}}, '
                             '         "slack":       {obj: {x, y, z_rot}}}. '
                             'Mutually exclusive with --subtask_constraints. '
                             'When set, slack values are world-absolute (NOT relative to seed).')

    args = parser.parse_args(argv)
    if args.chained_warp_constraint and args.subtask_constraints:
        parser.error(
            "--chained_warp_constraint and --subtask_constraints are mutually exclusive"
        )

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

    # --- Optional: subtask pose constraints (reject trials that miss the target) ---
    if args.subtask_constraints:
        import json as _json2
        from mimicgen.datagen.data_generator import DataGenerator

        subtask_constraints: dict[str, dict] = _json2.loads(args.subtask_constraints)

        # Build a helper that checks whether current object poses satisfy the constraint.
        # Re-uses the same world→relative coordinate transform as the IC constraint above.
        def _poses_satisfy_constraint(
            cur_poses: dict,
            constraint: dict,
            env_cls,
            seed_poses_for_sc: dict,
        ) -> bool:
            """Return True if all constrained objects are within the allowed range."""
            try:
                bounds = env_cls._get_initial_placement_bounds(None)  # type: ignore
            except Exception:
                return True  # Can't check — don't reject
            for obj_name, axes in constraint.items():
                matched_cur = next(
                    (v for k, v in cur_poses.items()
                     if k == obj_name or k in obj_name or obj_name in k),
                    None,
                )
                if matched_cur is None:
                    continue
                bounds_key = next(
                    (bk for bk in bounds
                     if bk == obj_name or bk in obj_name or obj_name in bk),
                    None,
                )
                if bounds_key is None:
                    continue
                seed_for_obj = next(
                    (s for sk, s in seed_poses_for_sc.items()
                     if sk == obj_name or sk in obj_name or obj_name in sk),
                    None,
                ) or {}

                for axis, allowed_range in axes.items():
                    if allowed_range is None:
                        continue
                    if axis not in ("x", "y", "z_rot"):
                        continue
                    if axis in ("x", "y"):
                        ref_idx = 0 if axis == "x" else 1
                        ref_val = float(bounds[bounds_key]["reference"][ref_idx])
                        cur_rel = matched_cur.get(axis, 0.0) - ref_val
                        seed_rel = seed_for_obj.get(axis, 0.0) - ref_val
                    else:
                        cur_rel = matched_cur.get(axis, 0.0)
                        seed_rel = seed_for_obj.get(axis, 0.0)
                    lo = seed_rel + allowed_range[0]
                    hi = seed_rel + allowed_range[1]
                    if not (lo <= cur_rel <= hi):
                        return False
            return True

        # Resolve env class (same logic as IC constraint above).
        import h5py as _h5py2
        with _h5py2.File(str(seed_hdf5), "r") as _f2:
            _env_meta2 = _json2.loads(_f2["data"].attrs["env_args"])
        _env_name2 = _env_meta2.get("env_name", "")
        import robosuite.environments as _renv2
        _env_cls2 = _renv2.REGISTERED_ENVS.get(_env_name2)

        # Seed poses to use as reference for constraint offsets: from IC constraint
        # block above when available, otherwise read fresh.
        if args.seed_object_poses:
            _sc_seed_poses = _json2.loads(args.seed_object_poses)
        else:
            _sc_seed_poses = {}

        if _env_cls2 is not None and subtask_constraints:
            _orig_dg_generate = DataGenerator.generate

            def _constrained_generate(self, env, env_interface, **kw):  # type: ignore[misc]
                """Wrap DataGenerator.generate to reject trials missing subtask constraints."""
                # Build an instrumented version: after each subtask executes, check.
                # We achieve this by counting subtask iterations with a mutable counter.
                subtask_counter = [0]
                _orig_exec = env_interface.__class__.get_datagen_info

                # We can't easily hook into the subtask loop from outside the method,
                # so we temporarily patch get_datagen_info to intercept calls made
                # from data_generator.py line 267 (cur_datagen_info = ...) which
                # happens BEFORE each subtask.  We check the PREVIOUS subtask's
                # constraint on the NEXT call (i.e., after execution completes).
                check_results = [True]  # [constraint_satisfied]
                prev_poses: dict = {}

                _orig_gdi = env_interface.__class__.get_datagen_info

                def _patched_gdi(iface):
                    info = _orig_gdi(iface)
                    si = subtask_counter[0]
                    # On subtask si > 0, check the constraint for subtask si-1
                    # (which just finished).
                    if si > 0 and check_results[0]:
                        constraint_key = str(si - 1)
                        if constraint_key in subtask_constraints:
                            ok = _poses_satisfy_constraint(
                                prev_poses,
                                subtask_constraints[constraint_key],
                                _env_cls2,
                                _sc_seed_poses,
                            )
                            if not ok:
                                check_results[0] = False
                    # Record current poses for the next check.
                    prev_poses.clear()
                    prev_poses.update({k: {"x": v[0], "y": v[1], "z_rot": v[2]}
                                       for k, v in (
                                           info.object_poses.items()
                                           if hasattr(info.object_poses, "items")
                                           else {}
                                       )})
                    subtask_counter[0] += 1
                    return info

                env_interface.__class__.get_datagen_info = _patched_gdi
                try:
                    result = _orig_dg_generate(self, env, env_interface, **kw)
                finally:
                    env_interface.__class__.get_datagen_info = _orig_gdi

                # If constraint was violated, mark the trial as failed.
                if not check_results[0]:
                    return {
                        "initial_state": result.get("initial_state"),
                        "states": [],
                        "observations": [],
                        "datagen_infos": [],
                        "actions": [],
                        "success": False,
                        "src_demo_inds": [],
                        "src_demo_labels": [],
                    }
                return result

            DataGenerator.generate = _constrained_generate  # type: ignore[method-assign]
            print(
                f"[run_mimicgen_generate] subtask constraints active: "
                + "; ".join(
                    f"after subtask {si}: {list(c.keys())}"
                    for si, c in subtask_constraints.items()
                )
            )

    # --- Optional: install the chained-warp data generator -----------------
    # Replaces the default mimicgen DataGenerator with a subclass that aborts
    # the trial early if the constrained subtask's object pose lands outside
    # the slack box. Outer trial-budget loop in generate_dataset() handles the
    # retry.
    if args.chained_warp_constraint:
        import json as _json3
        from policy_doctor.mimicgen.chained_warp_generator import (
            IntermediateConstraint,
            make_chained_warp_generator_class,
        )
        import mimicgen.datagen.data_generator as _dg_mod
        import mimicgen.scripts.generate_dataset as _gd_mod

        _cw = _json3.loads(args.chained_warp_constraint)
        constraint = IntermediateConstraint(
            subtask_idx=int(_cw["subtask_idx"]),
            target_pose=_cw["target_pose"],
            slack=_cw["slack"],
            slack_widen_factor=float(_cw.get("slack_widen_factor", 2.0)),
            objects=_cw.get("objects"),
        )
        _BaseDG = _dg_mod.DataGenerator
        _ChainedWarpClass = make_chained_warp_generator_class()
        # Pre-bind the constraint via a thin factory so the rest of MimicGen
        # can instantiate it just like the base DataGenerator.

        class _BoundCW(_ChainedWarpClass):
            def __init__(self, *a, **kw):
                super().__init__(*a, constraint=constraint, **kw)

        # Swap in both spots that import DataGenerator by name.
        _dg_mod.DataGenerator = _BoundCW  # type: ignore[assignment]
        if hasattr(_gd_mod, "DataGenerator"):
            _gd_mod.DataGenerator = _BoundCW  # type: ignore[assignment]

        print(
            f"[run_mimicgen_generate] chained-warp constraint installed: "
            f"subtask={constraint.subtask_idx}  "
            f"objects={list(constraint.target_pose.keys())}"
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
