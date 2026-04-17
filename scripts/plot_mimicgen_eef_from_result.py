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
    seed_xyz = np.array(seed_raw[0], dtype=np.float32) if seed_raw else None
    generated_eef = [np.array(t, dtype=np.float32) for t in gen_raw]

    stats = result.get("stats", {})
    n_gen = len(generated_eef)
    n_success = stats.get("num_success", n_gen)
    n_attempts = stats.get("num_attempts", "?")
    rate = stats.get("success_rate", "?")
    seed_key = result.get("seed_demo_key", "demo_0")

    subtitle = f"seed={seed_key}  |  n_gen={n_gen}  |  success={n_success}/{n_attempts} ({rate:.0f}%)"

    fig_3d = create_eef_trajectory_figure(
        seed_xyz=seed_xyz,
        generated_xyz_list=generated_eef,
        title=f"EEF Trajectories — Square D0  |  {subtitle}",
    )
    fig_2d = create_initial_eef_scatter_2d(
        seed_xyz=seed_xyz,
        generated_xyz_list=generated_eef,
        title=f"Initial EEF Positions (t=0)  |  {subtitle}",
    )

    out_3d = out_dir / "eef_3d.html"
    out_2d = out_dir / "eef_2d_initial.html"
    pio.write_html(fig_3d, str(out_3d))
    pio.write_html(fig_2d, str(out_2d))

    print(f"Plots written:")
    print(f"  3D trajectories : {out_3d}")
    print(f"  2D initial pos  : {out_2d}")


if __name__ == "__main__":
    main()
