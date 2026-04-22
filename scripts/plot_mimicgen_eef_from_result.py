"""Produce EEF trajectory HTML plots from a generate_mimicgen_demos result.json.

Run in the policy_doctor conda env (has plotly):
    python scripts/plot_mimicgen_eef_from_result.py --result /tmp/mimicgen_eef_test/result.json

Flags (all default True, pass --no-<flag> to disable):
    --no-nut-pose             hide square nut initial pose overlay
    --no-gripper-at-lowest-z  hide gripper orientation at lowest-Z overlay
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def _load_pose_list(raw: list) -> list:
    """Convert list of nested Python lists → list of (4,4) float32 ndarrays."""
    import numpy as np
    return [np.array(p, dtype=np.float32) for p in raw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="/tmp/mimicgen_eef_test/result.json")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: same as result)")
    ap.add_argument("--no-nut-pose", dest="show_nut_pose",
                    action="store_false", default=True,
                    help="Hide square nut initial pose overlay")
    ap.add_argument("--no-gripper-at-lowest-z", dest="show_gripper_at_lowest_z",
                    action="store_false", default=True,
                    help="Hide gripper orientation at lowest-Z overlay")
    args = ap.parse_args()

    result_path = Path(args.result)
    out_dir = Path(args.out_dir) if args.out_dir else result_path.parent

    with open(result_path) as f:
        result = json.load(f)

    import numpy as np
    import plotly.io as pio

    from policy_doctor.plotting.plotly.eef_trajectories import (
        create_eef_trajectory_figure,
        create_initial_eef_scatter_2d,
    )

    # --- EEF xyz trajectories ---
    seed_raw = result.get("seed_eef_xyz", [])
    seed_xyz = np.array(seed_raw[0], dtype=np.float32) if seed_raw else None
    success_eef = [np.array(t, dtype=np.float32) for t in result.get("generated_eef_xyz", [])]
    failed_eef  = [np.array(t, dtype=np.float32) for t in result.get("failed_eef_xyz", [])]

    # --- Pose data (optional; gracefully absent in older result.json) ---
    seed_pose_raw = result.get("seed_pose_at_lowest_z")
    seed_pose_at_lowest_z = np.array(seed_pose_raw, dtype=np.float32) if seed_pose_raw else None

    succ_poses_lz  = _load_pose_list(result.get("generated_eef_pose_at_lowest_z", []))
    fail_poses_lz  = _load_pose_list(result.get("failed_eef_pose_at_lowest_z", []))
    nut_t0_succ    = _load_pose_list(result.get("nut_poses_t0_success", []))
    nut_t0_fail    = _load_pose_list(result.get("nut_poses_t0_failed", []))
    nut_t0_all     = nut_t0_succ + nut_t0_fail

    # --- Subtitles ---
    stats = result.get("stats", {})
    n_succ = len(success_eef)
    n_fail = len(failed_eef)
    seed_key = result.get("seed_demo_key", "demo_0")
    sub_all  = f"seed={seed_key}  |  success={n_succ}  |  failed={n_fail}"
    sub_succ = f"seed={seed_key}  |  success={n_succ}"
    sub_fail = f"seed={seed_key}  |  failed={n_fail}"

    failed_arg = failed_eef if failed_eef else None

    show_nut  = args.show_nut_pose
    show_grip = args.show_gripper_at_lowest_z

    # --- Plot 1: all (success=green, failed=red) ---
    fig_3d = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=failed_arg,
        nut_poses_t0=nut_t0_all if show_nut else None,
        eef_poses_at_lowest_z=succ_poses_lz if show_grip else None,
        failed_eef_poses_at_lowest_z=fail_poses_lz if show_grip else None,
        seed_pose_at_lowest_z=seed_pose_at_lowest_z if show_grip else None,
        show_nut_pose=show_nut,
        show_gripper_at_lowest_z=show_grip,
        title=f"EEF Trajectories (all)  |  {sub_all}",
    )
    fig_2d = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=failed_arg,
        nut_poses_t0=nut_t0_all if show_nut else None,
        show_nut_pose=show_nut,
        title=f"Initial EEF Positions (t=0) — all  |  {sub_all}",
    )

    # --- Plot 2: success only ---
    fig_3d_succ = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=None,
        nut_poses_t0=nut_t0_succ if show_nut else None,
        eef_poses_at_lowest_z=succ_poses_lz if show_grip else None,
        seed_pose_at_lowest_z=seed_pose_at_lowest_z if show_grip else None,
        show_nut_pose=show_nut,
        show_gripper_at_lowest_z=show_grip,
        title=f"EEF Trajectories (success only)  |  {sub_succ}",
    )
    fig_2d_succ = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=None,
        nut_poses_t0=nut_t0_succ if show_nut else None,
        show_nut_pose=show_nut,
        title=f"Initial EEF Positions (t=0) — success only  |  {sub_succ}",
    )

    # --- Plot 3: failed only ---
    fig_3d_fail = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=failed_eef,
        failed_xyz_list=None,
        nut_poses_t0=nut_t0_fail if show_nut else None,
        eef_poses_at_lowest_z=fail_poses_lz if show_grip else None,
        seed_pose_at_lowest_z=seed_pose_at_lowest_z if show_grip else None,
        show_nut_pose=show_nut,
        show_gripper_at_lowest_z=show_grip,
        title=f"EEF Trajectories (failed only)  |  {sub_fail}",
    )
    fig_2d_fail = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=failed_eef,
        failed_xyz_list=None,
        nut_poses_t0=nut_t0_fail if show_nut else None,
        show_nut_pose=show_nut,
        title=f"Initial EEF Positions (t=0) — failed only  |  {sub_fail}",
    )

    pairs = [
        (fig_3d,      "eef_3d.html"),
        (fig_2d,      "eef_2d_initial.html"),
        (fig_3d_succ, "eef_3d_success.html"),
        (fig_2d_succ, "eef_2d_success.html"),
        (fig_3d_fail, "eef_3d_failed.html"),
        (fig_2d_fail, "eef_2d_failed.html"),
    ]
    print("Plots written:")
    for fig, name in pairs:
        p = out_dir / name
        pio.write_html(fig, str(p))
        print(f"  {p}")


if __name__ == "__main__":
    main()
