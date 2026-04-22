"""Quick test script: generate MimicGen demos from source/square.hdf5 and plot EEF trajectories.

Run in the mimicgen conda env:
    conda run -n mimicgen python scripts/test_mimicgen_eef_plots.py

Produces:
    /tmp/mimicgen_eef_test/eef_3d.html
    /tmp/mimicgen_eef_test/eef_2d_initial.html
    /tmp/mimicgen_eef_test/stats.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add repo root and vendored mimicgen to sys.path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
_MG = _REPO / "third_party" / "mimicgen"
if _MG.is_dir():
    sys.path.insert(0, str(_MG))

# Patch robomimic base_env shim (MimicGen compat)
try:
    import robomimic.envs.env_robosuite as er
    if not hasattr(er.EnvRobosuite, "base_env"):
        er.EnvRobosuite.base_env = property(lambda self: self.env)
except ImportError:
    pass

SOURCE_HDF5 = Path("/mnt/ssdB/erik/mimicgen_data/source/square.hdf5")
SEED_DEMO_KEY = "demo_0"
NUM_TRIALS = 50
OUT_DIR = Path("/tmp/mimicgen_eef_test")

# Force CPU for MuJoCo (headless); CUDA_VISIBLE_DEVICES handled by caller
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def run_generation(source_hdf5: Path, output_dir: Path, num_trials: int) -> dict:
    from mimicgen.configs import config_factory
    from mimicgen.scripts.generate_dataset import generate_dataset

    cfg = config_factory("square", "robosuite")
    cfg.experiment.name = "eef_test"
    cfg.experiment.source.dataset_path = str(source_hdf5)
    cfg.experiment.source.n = 1          # use only the first demo as seed
    cfg.experiment.source.start = 0
    cfg.experiment.generation.path = str(output_dir / "_gen")
    cfg.experiment.generation.num_trials = num_trials
    cfg.experiment.generation.guarantee = False
    cfg.experiment.generation.keep_failed = True   # preserve failures for success/failure colouring
    cfg.experiment.render_video = False
    cfg.experiment.num_demo_to_render = 0
    cfg.experiment.num_fail_demo_to_render = 0
    cfg.experiment.max_num_failures = num_trials * 3
    cfg.experiment.log_every_n_attempts = 25
    cfg.obs.collect_obs = True        # must be True; write_demo_to_hdf5 breaks on None obs
    cfg.obs.camera_names = []         # no image obs — low-dim state only

    print(f"[test] Generating {num_trials} trials from {source_hdf5.name}[{SEED_DEMO_KEY}] ...")
    stats = generate_dataset(cfg, auto_remove_exp=True, render=False, video_path=None)
    print(f"[test] Generation done: {stats.get('num_success',0)}/{stats.get('num_attempts',0)} successes")
    return stats or {}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = OUT_DIR / "_gen" / "eef_test"

    stats = run_generation(SOURCE_HDF5, OUT_DIR, NUM_TRIALS)

    generated_hdf5 = gen_dir / "demo.hdf5"
    if not generated_hdf5.exists():
        print(f"[test] ERROR: generated demo.hdf5 not found at {generated_hdf5}")
        sys.exit(1)

    # Copy to output dir for easier access
    shutil.copy2(generated_hdf5, OUT_DIR / "demo.hdf5")

    # Copy failed demos if present
    generated_failed = gen_dir / "demo_failed.hdf5"
    if generated_failed.exists():
        shutil.copy2(generated_failed, OUT_DIR / "demo_failed.hdf5")
        print(f"[test] demo_failed.hdf5 copied")

    # Save stats
    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"[test] stats saved to {stats_path}")

    # Extract EEF trajectories + pose data
    from policy_doctor.mimicgen.eef import (
        extract_eef_xyz_from_hdf5,
        extract_eef_pose_at_lowest_z_from_hdf5,
        extract_object_pose_t0_from_hdf5,
    )

    seed_eef = extract_eef_xyz_from_hdf5(SOURCE_HDF5)
    seed_xyz = seed_eef[0] if seed_eef else None
    if seed_xyz is not None:
        print(f"[test] seed EEF shape: {seed_xyz.shape}")
    else:
        print("[test] WARNING: no EEF found in source/seed HDF5")

    seed_pose_lowest_z_list = extract_eef_pose_at_lowest_z_from_hdf5(SOURCE_HDF5)
    seed_pose_at_lowest_z = seed_pose_lowest_z_list[0] if seed_pose_lowest_z_list else None

    generated_hdf5 = OUT_DIR / "demo.hdf5"
    generated_eef = extract_eef_xyz_from_hdf5(generated_hdf5)
    print(f"[test] generated EEF trajectories (success): {len(generated_eef)}")
    generated_pose_at_lowest_z = extract_eef_pose_at_lowest_z_from_hdf5(generated_hdf5)
    nut_poses_t0_succ = extract_object_pose_t0_from_hdf5(generated_hdf5, "square_nut")
    print(f"[test] nut poses (success): {len(nut_poses_t0_succ)}")

    failed_hdf5 = OUT_DIR / "demo_failed.hdf5"
    failed_eef: list = []
    failed_pose_at_lowest_z: list = []
    nut_poses_t0_fail: list = []
    if failed_hdf5.exists():
        failed_eef = extract_eef_xyz_from_hdf5(failed_hdf5)
        failed_pose_at_lowest_z = extract_eef_pose_at_lowest_z_from_hdf5(failed_hdf5)
        nut_poses_t0_fail = extract_object_pose_t0_from_hdf5(failed_hdf5, "square_nut")
        print(f"[test] generated EEF trajectories (failed): {len(failed_eef)}")
    else:
        print("[test] no demo_failed.hdf5 found")

    # Combine nut poses (all demos) — used for the "all" plot
    nut_poses_t0_all = nut_poses_t0_succ + nut_poses_t0_fail

    # Save EEF data as JSON (matches pipeline step result format)
    result = {
        "seed_demo_key": SEED_DEMO_KEY,
        "generated_hdf5_path": str(generated_hdf5),
        "stats": stats,
        "seed_eef_xyz": [seed_xyz.tolist()] if seed_xyz is not None else [],
        "generated_eef_xyz": [t.tolist() for t in generated_eef],
        "failed_eef_xyz": [t.tolist() for t in failed_eef],
        # Pose data (4×4 matrices serialised as nested lists)
        "seed_pose_at_lowest_z": seed_pose_at_lowest_z.tolist() if seed_pose_at_lowest_z is not None else None,
        "generated_eef_pose_at_lowest_z": [p.tolist() for p in generated_pose_at_lowest_z],
        "failed_eef_pose_at_lowest_z": [p.tolist() for p in failed_pose_at_lowest_z],
        "nut_poses_t0_success": [p.tolist() for p in nut_poses_t0_succ],
        "nut_poses_t0_failed": [p.tolist() for p in nut_poses_t0_fail],
    }
    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(result, f)
    print(f"[test] result.json saved to {OUT_DIR / 'result.json'}")
    print(f"[test] run plot_mimicgen_eef_from_result.py in policy_doctor env to produce HTML/PNG plots")


if __name__ == "__main__":
    main()
