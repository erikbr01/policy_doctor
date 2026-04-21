"""Produce EEF trajectory HTML plots from a generate_mimicgen_demos result.json.

Run in the policy_doctor conda env (has plotly):
    python scripts/plot_mimicgen_eef_from_result.py --result /tmp/mimicgen_eef_test/result.json
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="/tmp/mimicgen_eef_test/result.json")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: same as result)")
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

    seed_raw = result.get("seed_eef_xyz", [])
    gen_raw = result.get("generated_eef_xyz", [])
    fail_raw = result.get("failed_eef_xyz", [])
    seed_xyz = np.array(seed_raw[0], dtype=np.float32) if seed_raw else None
    success_eef = [np.array(t, dtype=np.float32) for t in gen_raw]
    failed_eef = [np.array(t, dtype=np.float32) for t in fail_raw]

    stats = result.get("stats", {})
    n_succ = len(success_eef)
    n_fail = len(failed_eef)
    n_attempts = stats.get("num_attempts", n_succ + n_fail)
    seed_key = result.get("seed_demo_key", "demo_0")

    subtitle_both = f"seed={seed_key}  |  success={n_succ}  |  failed={n_fail}"
    subtitle_succ = f"seed={seed_key}  |  success={n_succ}"
    subtitle_fail = f"seed={seed_key}  |  failed={n_fail}"

    failed_arg = failed_eef if failed_eef else None

    # --- Plot 1: successes + failures together (coloured) ---
    fig_3d = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=failed_arg,
        title=f"EEF Trajectories (all)  |  {subtitle_both}",
    )
    fig_2d = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=failed_arg,
        title=f"Initial EEF Positions (t=0) — all  |  {subtitle_both}",
    )

    # --- Plot 2: successes only ---
    fig_3d_succ = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=None,
        title=f"EEF Trajectories (success only)  |  {subtitle_succ}",
    )
    fig_2d_succ = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=success_eef,
        failed_xyz_list=None,
        title=f"Initial EEF Positions (t=0) — success only  |  {subtitle_succ}",
    )

    # --- Plot 3: failures only ---
    fig_3d_fail = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=failed_eef,
        failed_xyz_list=None,
        title=f"EEF Trajectories (failed only)  |  {subtitle_fail}",
    )
    fig_2d_fail = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=failed_eef,
        failed_xyz_list=None,
        title=f"Initial EEF Positions (t=0) — failed only  |  {subtitle_fail}",
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
