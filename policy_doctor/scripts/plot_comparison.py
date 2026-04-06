"""Plot training curve comparison: mean ± std across seeds, fetched from wandb.

Usage examples:
    # Compare baseline vs p95 (defaults for transport_mh):
    python -m policy_doctor.scripts.plot_comparison

    # Custom groups:
    python -m policy_doctor.scripts.plot_comparison \\
        --group "baseline:data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0,data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_1,data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_2" \\
        --group "p95:data/outputs/train/mar16/mar16_train_diffusion_unet_lowdim_transport_mh_0-curation_influence_sum_official-filter_0.5-select_0.0-p95,..."

    # Save to file:
    python -m policy_doctor.scripts.plot_comparison --output comparison.html
"""

import argparse
import pathlib

from policy_doctor.paths import REPO_ROOT

_REPO_ROOT = REPO_ROOT

_DEFAULT_GROUPS = [
    (
        "baseline (jan28)",
        [
            "data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_0",
            "data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_1",
            "data/outputs/train/jan28/jan28_train_diffusion_unet_lowdim_transport_mh_2",
        ],
    ),
    (
        "p95 curation (mar16)",
        [
            "data/outputs/train/mar16/mar16_train_diffusion_unet_lowdim_transport_mh_0-curation_influence_sum_official-filter_0.5-select_0.0-p95",
            "data/outputs/train/mar16/mar16_train_diffusion_unet_lowdim_transport_mh_1-curation_influence_sum_official-filter_0.5-select_0.0-p95",
            "data/outputs/train/mar16/mar16_train_diffusion_unet_lowdim_transport_mh_2-curation_influence_sum_official-filter_0.5-select_0.0-p95",
        ],
    ),
]


def parse_args():
    p = argparse.ArgumentParser(description="Compare training curves from wandb across experiment groups.")
    p.add_argument(
        "--group",
        action="append",
        metavar="LABEL:DIR1,DIR2,...",
        help="Add a group: 'label:dir1,dir2,...'. Can be repeated. Defaults to baseline vs p95.",
    )
    p.add_argument(
        "--metric", default="test/mean_score",
        help="Wandb metric key to plot (default: test/mean_score).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output path. .html for interactive, .png/.pdf for static. Defaults to stdout display.",
    )
    p.add_argument(
        "--n-bins", type=int, default=200,
        help="Number of interpolation bins for step alignment (default: 200).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    from policy_doctor.data.wandb_utils import load_curves_for_dirs, align_and_aggregate_seeds
    from policy_doctor.plotting.training_curves import create_training_comparison_plot

    if args.group:
        groups = []
        for spec in args.group:
            label, _, dirs_str = spec.partition(":")
            dirs = [d.strip() for d in dirs_str.split(",") if d.strip()]
            groups.append((label.strip(), dirs))
    else:
        groups = _DEFAULT_GROUPS

    series = []
    for label, dir_strs in groups:
        dirs = [_REPO_ROOT / d for d in dir_strs]
        missing = [d for d in dirs if not d.exists()]
        if missing:
            print(f"[warn] {label}: {len(missing)} dir(s) not found: {[str(m) for m in missing]}", file=sys.stderr)
        existing = [d for d in dirs if d.exists()]
        if not existing:
            print(f"[skip] {label}: no existing directories", file=sys.stderr)
            continue
        print(f"Fetching wandb history for '{label}' ({len(existing)} seeds)...", file=sys.stderr)
        curves, diagnostics = load_curves_for_dirs(existing, metric_key=args.metric)
        for diag in diagnostics:
            print(f"  [warn] {diag}", file=sys.stderr)
        if not curves:
            print(f"  [skip] {label}: no curves loaded", file=sys.stderr)
            continue
        steps, mean, std = align_and_aggregate_seeds(curves, n_bins=args.n_bins)
        series.append((label, steps, mean, std))
        print(f"  loaded {len(curves)} seed(s), {len(steps)} steps", file=sys.stderr)

    if not series:
        print("No data loaded. Check wandb connectivity and run directories.", file=sys.stderr)
        sys.exit(1)

    fig = create_training_comparison_plot(series, metric_key=args.metric)

    if args.output:
        out = pathlib.Path(args.output)
        if out.suffix == ".html":
            fig.write_html(str(out))
        else:
            fig.write_image(str(out))
        print(f"Saved to {out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
