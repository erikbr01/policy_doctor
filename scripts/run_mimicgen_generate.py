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
* ``demo.hdf5``  — generated demonstrations (merged)
* ``stats.json`` — generation statistics (success rate, counts, timing)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def _ensure_mimicgen_on_path() -> None:
    """Add vendored MimicGen to sys.path so ``import mimicgen`` resolves."""
    # Walk up from this script to find third_party/mimicgen
    this = Path(__file__).resolve()
    for parent in this.parents:
        candidate = parent / "third_party" / "mimicgen"
        if candidate.is_dir():
            s = str(candidate.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return
    # Fallback: assume already installed in the active env


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
    parser.add_argument("--seed_hdf5", required=True, help="Path to seed HDF5 (robomimic format)")
    parser.add_argument("--output_dir", required=True, help="Directory for generated output")
    parser.add_argument("--task_name", default="square", help="MimicGen task name (e.g. 'square')")
    parser.add_argument("--env_interface_name", default="MG_Square")
    parser.add_argument("--env_interface_type", default="robosuite")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--guarantee", action="store_true", default=False,
                        help="Keep attempting until num_trials successes (instead of attempts)")
    parser.add_argument(
        "--filter_key", default=None,
        help="Optional filter key for source dataset demo selection",
    )
    args = parser.parse_args(argv)

    _ensure_mimicgen_on_path()
    _apply_robomimic_base_env_shim()

    import mimicgen  # noqa: F401 — confirms env is correct
    from mimicgen.configs import config_factory
    from mimicgen.scripts.generate_dataset import generate_dataset
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset

    seed_hdf5 = Path(args.seed_hdf5).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: prepare_src_dataset (mutates seed_hdf5 in-place) ---
    print(f"[run_mimicgen_generate] prepare_src_dataset: {seed_hdf5}")
    prepare_src_dataset(
        dataset_path=str(seed_hdf5),
        env_interface_name=args.env_interface_name,
        env_interface_type=args.env_interface_type,
        filter_key=args.filter_key,
        n=None,
        output_path=None,  # in-place
    )
    print("[run_mimicgen_generate] prepare_src_dataset done.")

    # --- Step 2: generate_dataset ---
    # Use a temp sub-dir so generate_dataset can manage its own folder layout;
    # we copy demo.hdf5 to output_dir afterwards.
    gen_tmp = output_dir / "_gen_tmp"

    cfg = config_factory(args.task_name, "robosuite")
    cfg.experiment.name = "generated"
    cfg.experiment.source.dataset_path = str(seed_hdf5)
    cfg.experiment.source.n = None  # use all demos in the file
    cfg.experiment.generation.path = str(gen_tmp)
    cfg.experiment.generation.num_trials = args.num_trials
    cfg.experiment.generation.guarantee = args.guarantee
    cfg.experiment.generation.keep_failed = True   # preserve failures for success/failure colouring
    cfg.experiment.render_video = False
    cfg.experiment.num_demo_to_render = 0
    cfg.experiment.num_fail_demo_to_render = 0
    cfg.experiment.max_num_failures = max(args.num_trials * 2, 100)
    cfg.experiment.log_every_n_attempts = 100
    cfg.experiment.task.name = None   # auto-detected from env_meta
    cfg.obs.collect_obs = True        # must be True; write_demo_to_hdf5 crashes on None obs
    cfg.obs.camera_names = []         # no image obs — low-dim state only

    print(f"[run_mimicgen_generate] generate_dataset: num_trials={args.num_trials}")
    stats = generate_dataset(cfg, auto_remove_exp=True, render=False, video_path=None)

    # --- Step 3: copy demo.hdf5 + demo_failed.hdf5 and write stats ---
    generated_demo = gen_tmp / "generated" / "demo.hdf5"
    dest_demo = output_dir / "demo.hdf5"
    if generated_demo.is_file():
        shutil.copy2(generated_demo, dest_demo)
        print(f"[run_mimicgen_generate] demo.hdf5 written to {dest_demo}")
    else:
        print(f"[run_mimicgen_generate] WARNING: expected demo.hdf5 not found at {generated_demo}")

    generated_failed = gen_tmp / "generated" / "demo_failed.hdf5"
    dest_failed = output_dir / "demo_failed.hdf5"
    if generated_failed.is_file():
        shutil.copy2(generated_failed, dest_failed)
        print(f"[run_mimicgen_generate] demo_failed.hdf5 written to {dest_failed}")
    else:
        print("[run_mimicgen_generate] no demo_failed.hdf5 found (all trials succeeded?)")

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats or {}, f, indent=2, default=str)
    print(f"[run_mimicgen_generate] stats written to {stats_path}")

    # Clean up temp dir
    if gen_tmp.exists():
        shutil.rmtree(gen_tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
