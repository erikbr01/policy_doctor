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
    cfg.experiment.generation.keep_failed = False
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

    # Save stats
    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"[test] stats saved to {stats_path}")

    # Extract EEF trajectories
    from policy_doctor.mimicgen.eef import extract_eef_xyz_from_hdf5

    seed_eef = extract_eef_xyz_from_hdf5(SOURCE_HDF5)
    seed_xyz = seed_eef[0] if seed_eef else None
    if seed_xyz is not None:
        print(f"[test] seed EEF shape: {seed_xyz.shape}")
    else:
        print("[test] WARNING: no EEF found in source/seed HDF5")

    generated_eef = extract_eef_xyz_from_hdf5(OUT_DIR / "demo.hdf5")
    print(f"[test] generated EEF trajectories: {len(generated_eef)}")

    # Save EEF data as JSON (matches pipeline step result format)
    result = {
        "seed_demo_key": SEED_DEMO_KEY,
        "generated_hdf5_path": str(OUT_DIR / "demo.hdf5"),
        "stats": stats,
        "seed_eef_xyz": [seed_xyz.tolist()] if seed_xyz is not None else [],
        "generated_eef_xyz": [t.tolist() for t in generated_eef],
    }
    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(result, f)
    print(f"[test] result.json saved to {OUT_DIR / 'result.json'}")

    # Produce plots
    import plotly.io as pio
    from policy_doctor.plotting.plotly.eef_trajectories import (
        create_eef_trajectory_figure,
        create_initial_eef_scatter_2d,
    )

    fig_3d = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=generated_eef,
        title=f"EEF Trajectories — Square D0  |  seed={SEED_DEMO_KEY}  |  n_gen={len(generated_eef)}",
    )
    fig_2d = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=generated_eef,
        title=f"Initial EEF Positions (t=0)  |  seed={SEED_DEMO_KEY}  |  n_gen={len(generated_eef)}",
    )

    out_3d = OUT_DIR / "eef_3d.html"
    out_2d = OUT_DIR / "eef_2d_initial.html"
    pio.write_html(fig_3d, str(out_3d))
    pio.write_html(fig_2d, str(out_2d))
    print(f"\n[test] Plots written:")
    print(f"  3D trajectories : {out_3d}")
    print(f"  2D initial pos  : {out_2d}")
    print(f"\nOpen in browser:")
    print(f"  firefox {out_3d} &")
    print(f"  firefox {out_2d} &")


if __name__ == "__main__":
    main()
