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


def _patch_robomimic_api_compat() -> None:
    """Drop kwargs unsupported in robomimic 0.3.0 from create_env_for_data_processing.

    third_party/mimicgen passes env_class, render, etc. which robomimic 0.3.0 doesn't accept.
    """
    import inspect
    try:
        import robomimic.utils.env_utils as EnvUtils
    except ImportError:
        return
    sig = inspect.signature(EnvUtils.create_env_for_data_processing)
    if "env_class" in sig.parameters:
        return  # new API; no patch needed
    _orig = EnvUtils.create_env_for_data_processing

    def _patched(**kwargs):
        for k in list(kwargs.keys()):
            if k not in sig.parameters:
                kwargs.pop(k)
        return _orig(**kwargs)

    EnvUtils.create_env_for_data_processing = _patched


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
    parser.add_argument("--fix_initial_object_poses", action="store_true", default=False,
                        help="If set, constrain each object's initial pose relative to the "
                             "seed demo's pose (read from datagen_info after prepare_src_dataset).")
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
    _patch_robomimic_api_compat()

    import mimicgen  # noqa: F401
    from mimicgen.configs import config_factory
    from mimicgen.scripts.generate_dataset import generate_dataset
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

    seed_hdf5 = Path(args.seed_hdf5).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 0: inject model_file if missing (policy rollouts don't capture it) ---
    # Borrow model_file from the source demo HDF5 for this task, which always has
    # the correct MimicGen/MuJoCo 2.3.x model XML. The XML is purely structural —
    # it doesn't encode the seed trajectory state — so sharing it across seeds is safe.
    import json as _json
    import h5py as _h5py
    import os as _os
    _needs_model_file = False
    with _h5py.File(str(seed_hdf5), "r") as _f:
        for _k in _f.get("data", {}).keys():
            if _k.startswith("demo_") and "model_file" not in _f[f"data/{_k}"].attrs:
                _needs_model_file = True
                break
    if _needs_model_file:
        # Derive source dataset path from env_meta task name
        with _h5py.File(str(seed_hdf5), "r") as _f:
            _env_meta_str = _f["data"].attrs.get("env_args", "{}")
        _env_name = _json.loads(_env_meta_str).get("env_name", "").lower().replace("_", "")
        # Look for source dataset in standard locations
        _source_candidates = [
            f"/home/erbauer/data/mimicgen_data/core_datasets/core/{args.task_name}_d1.hdf5",
            f"/home/erbauer/data/mimicgen_data/source/{args.task_name}.hdf5",
        ]
        _model_xml = ""
        for _src_path in _source_candidates:
            if _os.path.exists(_src_path):
                with _h5py.File(_src_path, "r") as _sf:
                    _demo0 = sorted(k for k in _sf["data"].keys() if k.startswith("demo_"))[0]
                    if "model_file" in _sf[f"data/{_demo0}"].attrs:
                        _model_xml = _sf[f"data/{_demo0}"].attrs["model_file"]
                        print(f"[run_mimicgen_generate] model_file sourced from {_src_path} ({len(_model_xml)} chars)")
                        break
        if not _model_xml:
            print("[run_mimicgen_generate] WARNING: could not find model_file source; generation may fail")
        with _h5py.File(str(seed_hdf5), "a") as _f:
            for _k in list(_f.get("data", {}).keys()):
                if _k.startswith("demo_"):
                    _f[f"data/{_k}"].attrs["model_file"] = _model_xml

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

    # --- Filter seed demos with missing subtask termination signals ---
    # After prepare_src_dataset adds datagen_info, some demos may have subtask_term_signals
    # with sum=0 (the signal never fires), causing DataGenerator to assert-fail later.
    # Remove those demos now to avoid a cryptic assertion error downstream.
    import h5py as _h5py_filter
    import numpy as _np_filter
    with _h5py_filter.File(str(seed_hdf5), "r") as _f:
        _demos = sorted(k for k in _f["data"].keys() if k.startswith("demo_"))
        _bad = []
        for _d in _demos:
            _grp = _f["data"][_d]
            if "datagen_info" in _grp and "subtask_term_signals" in _grp["datagen_info"]:
                _sig = _grp["datagen_info"]["subtask_term_signals"]
                for _s in _sig.keys():
                    if _sig[_s][:].sum() == 0:
                        _bad.append(_d)
                        print(f"[run_mimicgen_generate] filtering demo {_d}: {_s}=0 (never fires)")
                        break
    if _bad:
        _good = [d for d in _demos if d not in set(_bad)]
        if not _good:
            raise RuntimeError(
                f"[run_mimicgen_generate] ALL seed demos have missing subtask signals; "
                f"cannot generate. Seed HDF5: {seed_hdf5}"
            )
        print(f"[run_mimicgen_generate] removed {len(_bad)} demos with missing signals, "
              f"{len(_good)} remain")
        with _h5py_filter.File(str(seed_hdf5), "a") as _f:
            for _d in _bad:
                del _f["data"][_d]
            _remaining = sorted(k for k in _f["data"].keys() if k.startswith("demo_"))
            for _i, _old in enumerate(_remaining):
                if _old != f"demo_{_i}":
                    _f["data"][f"demo_{_i}"] = _f["data"][_old]
                    del _f["data"][_old]

    # --- Optional: constrain object initial poses relative to seed ---
    if args.fix_initial_object_poses:
        import json as _json
        import h5py as _h5py
        import numpy as _np
        import robosuite.environments as _renv

        # Read seed poses from the now-prepared HDF5.  datagen_info was added by
        # prepare_src_dataset above, so these poses reflect the actual seed trajectory
        # — correct even when the seed comes from a policy rollout rather than a
        # source dataset demo (rollout HDF5 had no datagen_info before preparation).
        seed_poses: dict[str, dict[str, float]] = {}
        with _h5py.File(str(seed_hdf5), "r") as _f:
            _demo_keys = sorted(k for k in _f["data"].keys() if k.startswith("demo_"))
            if _demo_keys:
                _poses_grp_key = f"data/{_demo_keys[0]}/datagen_info/object_poses"
                if _poses_grp_key in _f:
                    for _obj_name in _f[_poses_grp_key].keys():
                        _poses = _np.array(_f[f"{_poses_grp_key}/{_obj_name}"])  # (T, 4, 4)
                        _pos = _poses[0, :3, 3]
                        _R = _poses[0, :3, :3]
                        seed_poses[_obj_name] = {
                            "x": float(_pos[0]),
                            "y": float(_pos[1]),
                            "z_rot": float(_np.arctan2(_R[1, 0], _R[0, 0])),
                        }
                else:
                    print("[run_mimicgen_generate] WARNING: datagen_info/object_poses not found "
                          "in prepared seed HDF5; skipping pose constraint")

        if not seed_poses:
            print("[run_mimicgen_generate] WARNING: no seed poses found; skipping pose constraint")
        else:
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

    # --- Build task spec: square uses a custom nn spec; all other tasks use the template ---
    _task_name_base = args.task_name.split("_")[0]  # "square", "coffee", etc.
    if _task_name_base == "square":
        # Square: subtask_1 uses nearest-neighbour with nn_k; subtask_2 is fixed peg (random).
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
    else:
        # All other tasks: use the template subtask spec (loaded via config_factory above).
        # Subtasks are plain Python dicts — use dict operations, not attribute access.
        # Only override dynamic parameters; keep object_ref, signals, and strategies from template.
        for subtask_key in list(cfg.task.task_spec.keys()):
            sub = cfg.task.task_spec[subtask_key]
            if "action_noise" in sub:
                sub["action_noise"] = args.action_noise
            if "num_interpolation_steps" in sub:
                sub["num_interpolation_steps"] = args.num_interpolation_steps
            if "num_fixed_steps" in sub:
                sub["num_fixed_steps"] = args.num_fixed_steps
            # Override nn_k for nearest-neighbour subtasks
            if sub.get("selection_strategy") == "nearest_neighbor_object":
                if sub.get("selection_strategy_kwargs") is None:
                    sub["selection_strategy_kwargs"] = {}
                sub["selection_strategy_kwargs"]["nn_k"] = args.nn_k
            # Override subtask_term_offset_range for subtasks that have it set (not None)
            if (
                sub.get("subtask_term_offset_range") is not None
                and (args.subtask_term_offset_lo != 0 or args.subtask_term_offset_hi != 0)
            ):
                sub["subtask_term_offset_range"] = (args.subtask_term_offset_lo, args.subtask_term_offset_hi)

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
