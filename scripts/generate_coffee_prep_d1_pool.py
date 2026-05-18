"""Generate CoffeePreparation_D1 pool from source demos.

Creates a large pool of D1-difficulty demos from the 10 source coffee preparation
demos. The pool is used as baseline training data for the may18 experiment.

The source dataset uses CoffeePreparation_D0 (fixed placements). This script
patches the env_name to CoffeePreparation_D1 to enable D1 randomization.

Usage (run in mimicgen conda env):
    conda run -n mimicgen python scripts/generate_coffee_prep_d1_pool.py \\
        --output_dir data/source/mimicgen/core_datasets/coffee_preparation_d1 \\
        --n_success 300 \\
        --probe   # probe mode: only 50 trials per demo to estimate success rate

Output:
    <output_dir>/demo.hdf5      — merged successful demos
    <output_dir>/stats.json     — per-demo-seed generation statistics
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _ensure_mimicgen_on_path() -> None:
    this = Path(__file__).resolve()
    for parent in this.parents:
        candidate = parent / "third_party" / "mimicgen"
        if candidate.is_dir():
            s = str(candidate.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            break


def _ensure_robosuite_shim() -> None:
    try:
        import robomimic.envs.env_robosuite as er
    except ImportError:
        return
    if not hasattr(er.EnvRobosuite, "base_env"):
        er.EnvRobosuite.base_env = property(lambda self: self.env)


def _patch_robomimic_api_compat() -> None:
    """Patch create_env_for_data_processing to drop kwargs unsupported in robomimic 0.3.0.

    The local third_party/mimicgen code was written for a newer robomimic that accepts
    env_class, render, render_offscreen, use_image_obs, use_depth_obs kwargs.
    robomimic 0.3.0 (in mimicgen_torch2) only accepts the base 5 args.
    """
    import inspect
    try:
        import robomimic.utils.env_utils as EnvUtils
    except ImportError:
        return
    sig = inspect.signature(EnvUtils.create_env_for_data_processing)
    if "env_class" in sig.parameters:
        return  # new API already supports env_class; no patch needed
    _orig = EnvUtils.create_env_for_data_processing
    _unsupported = {"env_class", "render", "render_offscreen", "use_image_obs", "use_depth_obs"}

    def _patched(**kwargs):
        for k in list(kwargs.keys()):
            if k not in sig.parameters:
                kwargs.pop(k)
        return _orig(**kwargs)

    EnvUtils.create_env_for_data_processing = _patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_hdf5",
        default=None,
        help="Path to source coffee_preparation.hdf5 (default: auto-detect from repo root)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/source/mimicgen/core_datasets/coffee_preparation_d1",
        help="Output directory for generated demos",
    )
    parser.add_argument(
        "--n_success",
        type=int,
        default=300,
        help="Target number of successful demos (with guarantee=True)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        default=False,
        help="Run probe mode: N trials for demo_0 only, report success rate, then exit",
    )
    parser.add_argument(
        "--n_trials_probe",
        type=int,
        default=50,
        help="Number of trials per source demo in probe mode",
    )
    parser.add_argument(
        "--n_trials_per_demo",
        type=int,
        default=None,
        help="Trials per source demo in full generation mode (default: 150). "
             "Generation stops early once n_success demos are collected.",
    )
    parser.add_argument(
        "--mug_zrot_range",
        type=float,
        default=None,
        help="Constrain mug z_rot to ±VALUE radians around seed pose (default: full D1 range = null). "
             "Use e.g. 1.571 for ±90°, 0.524 for ±30°.",
    )
    parser.add_argument(
        "--select_src_per_subtask",
        action="store_true",
        default=False,
        help="Allow each subtask to independently select its best source segment (may improve success rate).",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    _ensure_mimicgen_on_path()
    _ensure_robosuite_shim()
    _patch_robomimic_api_compat()

    import h5py
    import numpy as np
    import mimicgen  # noqa: F401 — registers env classes
    from mimicgen.configs import config_factory
    from mimicgen.scripts.generate_dataset import generate_dataset
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

    # --- Locate source HDF5 ---
    if args.source_hdf5:
        source_hdf5 = Path(args.source_hdf5).resolve()
    else:
        # Auto-detect: walk up from this script to find repo root
        this = Path(__file__).resolve()
        for parent in this.parents:
            candidate = parent / "third_party" / "cupid" / "data" / "source" / "mimicgen" / "source" / "coffee_preparation.hdf5"
            if candidate.is_file():
                source_hdf5 = candidate
                break
        else:
            raise FileNotFoundError(
                "Could not auto-detect source coffee_preparation.hdf5. "
                "Pass --source_hdf5 explicitly."
            )
    print(f"[generate_coffee_prep_d1_pool] source: {source_hdf5}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        # Resolve relative to repo root (third_party/cupid)
        this = Path(__file__).resolve()
        for parent in this.parents:
            if (parent / "third_party" / "cupid").is_dir():
                output_dir = parent / "third_party" / "cupid" / output_dir
                break
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- List source demos ---
    with h5py.File(source_hdf5, "r") as f:
        demo_keys = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
    print(f"[generate_coffee_prep_d1_pool] found {len(demo_keys)} source demos: {demo_keys}")

    all_stats = {}
    merged_demo_count = 0

    with tempfile.TemporaryDirectory(prefix="coffee_prep_d1_gen_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Probe or full generation per source demo
        if args.probe:
            n_trials = args.n_trials_probe
        else:
            n_trials = args.n_trials_per_demo if args.n_trials_per_demo is not None else 150

        for demo_key in demo_keys:
            print(f"\n[generate_coffee_prep_d1_pool] === Processing {demo_key} ===")

            # --- Copy and patch: D0 → D1 ---
            patched_hdf5 = tmpdir / f"seed_{demo_key}.hdf5"
            shutil.copy2(source_hdf5, patched_hdf5)
            with h5py.File(patched_hdf5, "r+") as f:
                env_args = json.loads(f["data"].attrs["env_args"])
                original_env = env_args.get("env_name", "?")
                env_args["env_name"] = "CoffeePreparation_D1"
                f["data"].attrs["env_args"] = json.dumps(env_args)
                # Remove all demos except the target key so MimicGen uses only this seed
                to_delete = [k for k in f["data"].keys() if k.startswith("demo_") and k != demo_key]
                for k in to_delete:
                    del f["data"][k]
                # Rename remaining demo to demo_0 if needed
                if demo_key != "demo_0" and demo_key in f["data"]:
                    f["data"][demo_key].name  # access to check
                    f.copy(f"data/{demo_key}", f["data"], "demo_0")
                    del f["data"][demo_key]
            print(f"  patched env_name: {original_env} → CoffeePreparation_D1")

            # --- prepare_src_dataset ---
            print(f"  prepare_src_dataset ...")
            prepare_src_dataset(
                dataset_path=str(patched_hdf5),
                env_interface_name="MG_CoffeePreparation",
                env_interface_type="robosuite",
                filter_key=None,
                n=None,
                output_path=None,
            )

            # --- Build MimicGen config from template ---
            gen_tmp = tmpdir / f"gen_{demo_key}"
            cfg = config_factory("coffee_preparation", "robosuite")
            cfg.experiment.name = "generated"
            cfg.experiment.source.dataset_path = str(patched_hdf5)
            cfg.experiment.source.n = None
            cfg.experiment.generation.path = str(gen_tmp)
            cfg.experiment.generation.num_trials = n_trials
            cfg.experiment.generation.guarantee = False  # fixed trials per demo; stop early via merged_demo_count check
            cfg.experiment.generation.keep_failed = False
            cfg.experiment.generation.interpolate_from_last_target_pose = True
            cfg.experiment.generation.transform_first_robot_pose = False
            cfg.experiment.generation.select_src_per_subtask = args.select_src_per_subtask
            cfg.experiment.render_video = False
            cfg.experiment.num_demo_to_render = 0
            cfg.experiment.num_fail_demo_to_render = 0
            cfg.experiment.max_num_failures = n_trials * 3
            cfg.experiment.log_every_n_attempts = 50
            cfg.experiment.task.name = None
            cfg.obs.collect_obs = True
            cfg.obs.camera_names = []

            # Override dynamic params in each subtask (keep object_ref, signals from template)
            action_noise = 0.05
            num_interpolation_steps = 5
            subtask_term_offset_range = (5, 10)

            for subtask_key in list(cfg.task.task_spec.keys()):
                sub = cfg.task.task_spec[subtask_key]
                if "action_noise" in sub:
                    sub["action_noise"] = action_noise
                if "num_interpolation_steps" in sub:
                    sub["num_interpolation_steps"] = num_interpolation_steps
                if "num_fixed_steps" in sub:
                    sub["num_fixed_steps"] = 0
                if "subtask_term_offset_range" in sub and sub["subtask_term_offset_range"] is not None:
                    sub["subtask_term_offset_range"] = subtask_term_offset_range

            # --- Optionally constrain mug z_rot for better retargetability ---
            if args.mug_zrot_range is not None:
                import numpy as _np
                import robosuite.environments as _renv
                with h5py.File(str(patched_hdf5), "r") as _f:
                    _env_args = json.loads(_f["data"].attrs["env_args"])
                env_name = _env_args.get("env_name", "CoffeePreparation_D1")
                env_cls = _renv.REGISTERED_ENVS.get(env_name)
                if env_cls is not None and hasattr(env_cls, "_get_initial_placement_bounds"):
                    _orig_bounds = env_cls._get_initial_placement_bounds
                    _zrot_range = float(args.mug_zrot_range)

                    def _constrained_bounds(self, _orig=_orig_bounds, _r=_zrot_range):
                        bounds = _orig(self)
                        if "mug" in bounds:
                            lo, hi = bounds["mug"]["z_rot"]
                            seed_zrot = (lo + hi) / 2
                            bounds["mug"]["z_rot"] = (seed_zrot - _r, seed_zrot + _r)
                        return bounds

                    env_cls._get_initial_placement_bounds = _constrained_bounds
                    print(f"  mug z_rot constrained to ±{_np.degrees(_zrot_range):.0f}°")

            # --- Generate ---
            print(f"  generate_dataset: n_trials={n_trials}, guarantee={cfg.experiment.generation.guarantee}")
            stats = generate_dataset(cfg, auto_remove_exp=True, render=False, video_path=None)
            all_stats[demo_key] = stats or {}

            successes = (stats or {}).get("num_success", 0)
            total_attempts = (stats or {}).get("num_attempts", n_trials)
            rate = successes / max(total_attempts, 1)
            print(f"  result: {successes} successes / {total_attempts} attempts = {rate:.1%}")

            # --- Collect output ---
            demo_hdf5 = gen_tmp / "generated" / "demo.hdf5"
            if demo_hdf5.is_file() and successes > 0:
                # Copy demos to a per-seed subdirectory of output
                seed_out = output_dir / f"seed_{demo_key}"
                seed_out.mkdir(exist_ok=True)
                shutil.copy2(demo_hdf5, seed_out / "demo.hdf5")
                merged_demo_count += successes
                print(f"  saved {successes} demos → {seed_out}/demo.hdf5")

            if args.probe:
                print(f"\n[generate_coffee_prep_d1_pool] PROBE RESULT for {demo_key}:")
                print(f"  Success rate: {rate:.1%} ({successes}/{total_attempts})")
                print(f"  Estimated trials for 300 demos: {int(300 / max(rate, 0.001))}")
                continue  # probe: only run one demo then report

            # Early termination once we have enough
            if merged_demo_count >= args.n_success:
                print(f"\n[generate_coffee_prep_d1_pool] Reached target ({args.n_success}) — stopping early")
                break

    # --- Write stats summary ---
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"\n[generate_coffee_prep_d1_pool] stats → {stats_path}")

    if args.probe:
        print(f"\n[generate_coffee_prep_d1_pool] PROBE COMPLETE. Check stats above.")
        print(f"If success rate is adequate, rerun without --probe to generate full pool.")
        return

    # --- Merge all per-seed demo HDF5s into one ---
    print(f"\n[generate_coffee_prep_d1_pool] Merging {merged_demo_count} demos ...")
    _merge_demo_hdf5s(output_dir, output_dir / "demo.hdf5")
    print(f"[generate_coffee_prep_d1_pool] DONE. Merged HDF5: {output_dir}/demo.hdf5")
    print(f"Total demos: {merged_demo_count}")


def _merge_demo_hdf5s(source_dir: Path, output_path: Path) -> None:
    """Merge all demo.hdf5 files in subdirectories into a single HDF5."""
    import h5py

    demo_files = sorted(source_dir.glob("seed_*/demo.hdf5"))
    if not demo_files:
        print("[merge] No demo files found.")
        return

    global_idx = 0
    with h5py.File(output_path, "w") as out_f:
        data_grp = out_f.require_group("data")

        # Copy env_args from first file
        with h5py.File(demo_files[0], "r") as first_f:
            for attr_name, attr_val in first_f["data"].attrs.items():
                data_grp.attrs[attr_name] = attr_val

        for demo_file in demo_files:
            with h5py.File(demo_file, "r") as in_f:
                src_keys = sorted(k for k in in_f["data"].keys() if k.startswith("demo_"))
                for src_key in src_keys:
                    dst_key = f"demo_{global_idx}"
                    in_f.copy(f"data/{src_key}", data_grp, name=dst_key)
                    global_idx += 1

        data_grp.attrs["total"] = global_idx
    print(f"[merge] Merged {global_idx} demos → {output_path}")


if __name__ == "__main__":
    main()
