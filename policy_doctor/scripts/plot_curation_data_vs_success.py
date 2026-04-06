"""CLI: scatter plot of training data size vs mean training rollout success."""

from __future__ import annotations

import argparse
import pathlib

from policy_doctor.paths import REPO_ROOT

_REPO_ROOT = REPO_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scatter: training dataset size vs training-log test/mean_score. "
        "Uses the last checkpoint per run for dataset size; emits one point per eval in the "
        "chronological last-K lines of logs.json.txt (boxplots pool those y-values)."
    )
    p.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=_REPO_ROOT,
        help="Repository root (default: cupid root).",
    )
    p.add_argument(
        "--train-root",
        type=pathlib.Path,
        default=None,
        help="Override train output root (default: <repo>/data/outputs/train).",
    )
    p.add_argument(
        "--eval-root",
        type=pathlib.Path,
        default=None,
        help="eval_save_episodes root for optional offline hover (default: <repo>/data/outputs/eval_save_episodes).",
    )
    p.add_argument("--task-substring", default="transport_mh", help="Substring in run directory name.")
    p.add_argument(
        "--policy-substring",
        default="diffusion_unet_lowdim",
        help="Substring in run directory name (policy).",
    )
    p.add_argument(
        "--baseline-train-date",
        default="jan28",
        help="Train date folder name marking baseline runs (no -curation in name).",
    )
    p.add_argument(
        "--rollout-window",
        type=int,
        default=5,
        help="Take this many trailing test/mean_score rows from logs.json.txt per run (default: 5); "
        "each row is its own plotted y (not averaged).",
    )
    p.add_argument(
        "--no-offline-eval",
        action="store_true",
        help="Do not look for eval_log.json under eval_save_episodes for hover.",
    )
    p.add_argument(
        "--include-policy-doctor-curation-configs",
        action="store_true",
        help=(
            "Include curated runs whose sample_curation_config lives under policy_doctor/policy_doctor/configs. "
            "Default: only runs using influence_visualizer/configs (jan28-style baselines are always included)."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=None,
        help="Write figure (.html, .png, .pdf, .svg). Default: show in browser.",
    )
    p.add_argument("--title", default=None, help="Plot title.")
    p.add_argument(
        "--boxplot-train-name",
        default=None,
        metavar="NAME",
        help=(
            "Any training run directory name in the experiment; used to derive the experiment key "
            "(all seeds; last-K log eval y-values in that experiment are pooled in the boxplot). "
            "Example: jan28_train_diffusion_unet_lowdim_transport_mh_0."
        ),
    )
    p.add_argument(
        "--boxplot-experiment-key",
        default=None,
        metavar="KEY",
        help=(
            "Explicit experiment key (same grouping as experiment_key_from_train_name). "
            "If set, overrides derivation from --boxplot-train-name."
        ),
    )
    p.add_argument(
        "--boxplot-output",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help="Path for boxplot (.html, .png, ...). Default: <main_output_stem>_boxplot<suffix> if -o is set.",
    )
    p.add_argument(
        "--boxplot-title",
        default=None,
        help="Title for the boxplot figure only.",
    )
    p.add_argument(
        "--boxplot-all-experiments",
        action="store_true",
        help=(
            "Write one figure with a box per distinct experiment (same grouping as "
            "experiment_key_from_train_name), pooling all seeds and last-K log evals each."
        ),
    )
    p.add_argument(
        "--boxplot-experiments",
        default=None,
        metavar="K1,K2,...",
        help=(
            "Comma-separated experiment keys for a multi-box figure (subset you choose). "
            "If combined with --boxplot-all-experiments, only keys in this list are kept (must still appear in the scan)."
        ),
    )
    p.add_argument(
        "--matplotlib-pdf",
        action="store_true",
        help=(
            "Also write Matplotlib vector PDFs next to --output: "
            "<stem>_mpl_scatter.pdf and (with multi boxplot) <stem>_mpl_boxplots_multi.pdf."
        ),
    )
    p.add_argument(
        "--scatter-x",
        choices=("mean_score_order", "data_size"),
        default="mean_score_order",
        help=(
            "Scatter x-axis: mean_score_order (default) = experiment columns by ascending mean "
            "log score, spread ∝ train seq count, baseline column on the right; data_size = raw "
            "sequence count on x."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    from policy_doctor.data.curation_eval_scan import (
        collect_curation_scatter_points,
        experiment_key_from_train_name,
    )
    from policy_doctor.plotting.curation_scatter import (
        create_curation_data_vs_success_scatter,
        create_experiment_checkpoint_score_boxplot,
        create_multi_experiment_checkpoint_score_boxplots,
    )

    points = collect_curation_scatter_points(
        repo_root=repo_root,
        train_root=args.train_root.resolve() if args.train_root else None,
        eval_save_root=args.eval_root.resolve() if args.eval_root else None,
        task_substring=args.task_substring,
        policy_substring=args.policy_substring,
        baseline_train_date=args.baseline_train_date,
        rollout_window=args.rollout_window,
        load_offline_eval=not args.no_offline_eval,
        only_influence_visualizer_configs=not args.include_policy_doctor_curation_configs,
    )

    if not points:
        print("No points collected. Check --train-root and that runs have logs.json.txt + checkpoints.", file=sys.stderr)
        sys.exit(1)

    title = args.title or (
        f"{args.task_substring}: data size vs log test/mean_score "
        f"(last {args.rollout_window} evals per run, one point each)"
    )
    fig = create_curation_data_vs_success_scatter(
        points,
        title=title,
        rollout_window=args.rollout_window,
        baseline_train_date=args.baseline_train_date,
        task_substring=args.task_substring,
        scatter_x=args.scatter_x,
    )

    any_multi = bool(args.boxplot_all_experiments or args.boxplot_experiments)
    any_single = bool(args.boxplot_train_name or args.boxplot_experiment_key)
    if any_multi and any_single:
        print(
            "Use either single-experiment boxplot (--boxplot-train-name / --boxplot-experiment-key) "
            "or multi (--boxplot-all-experiments / --boxplot-experiments), not both.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.output:
        out = args.output
        if out.suffix.lower() == ".html":
            fig.write_html(str(out))
        else:
            fig.write_image(str(out))
        print(f"Wrote {out}")
        if args.matplotlib_pdf:
            from policy_doctor.plotting.curation_scatter_mpl import (
                save_curation_scatter_matplotlib_pdf,
            )

            scatter_pdf = out.resolve().parent / f"{out.stem}_mpl_scatter.pdf"
            save_curation_scatter_matplotlib_pdf(
                scatter_pdf,
                points,
                title=title,
                rollout_window=args.rollout_window,
                baseline_train_date=args.baseline_train_date,
                task_substring=args.task_substring,
                scatter_x=args.scatter_x,
            )
            print(f"Wrote {scatter_pdf}")
    elif not any_single and not any_multi:
        fig.show()
    elif args.matplotlib_pdf:
        print("--matplotlib-pdf requires --output / -o (for PDF path stem).", file=sys.stderr)
        sys.exit(2)

    if any_single:
        if args.boxplot_experiment_key:
            exp_key = args.boxplot_experiment_key
        else:
            exp_key = experiment_key_from_train_name(args.boxplot_train_name, args.task_substring)

        box_sub = [
            p
            for p in points
            if experiment_key_from_train_name(p.train_name, args.task_substring) == exp_key
        ]
        if not box_sub:
            keys = sorted({experiment_key_from_train_name(p.train_name, args.task_substring) for p in points})
            preview = "\n  ".join(keys[:30])
            more = f"\n  ... ({len(keys) - 30} more)" if len(keys) > 30 else ""
            print(
                f"No collected points for experiment key {exp_key!r} "
                f"(from --boxplot-experiment-key or --boxplot-train-name).\n"
                f"Sample experiment keys from this scan:\n  {preview}{more}",
                file=sys.stderr,
            )
            sys.exit(1)

        box_out = args.boxplot_output
        if box_out is None:
            if args.output:
                box_out = args.output.parent / f"{args.output.stem}_boxplot{args.output.suffix}"
            else:
                print(
                    "Boxplot needs a destination: pass --boxplot-output or main -o "
                    "(boxplot will be written next to scatter).",
                    file=sys.stderr,
                )
                sys.exit(2)

        fig_box = create_experiment_checkpoint_score_boxplot(
            points,
            experiment_key=exp_key,
            task_substring=args.task_substring,
            rollout_window=args.rollout_window,
            title=args.boxplot_title,
        )
        box_out = box_out.resolve()
        if box_out.suffix.lower() == ".html":
            fig_box.write_html(str(box_out))
        else:
            fig_box.write_image(str(box_out))
        print(f"Wrote {box_out}")

    if any_multi:
        all_keys = sorted(
            {experiment_key_from_train_name(p.train_name, args.task_substring) for p in points}
        )
        if args.boxplot_experiments:
            want = [k.strip() for k in args.boxplot_experiments.split(",") if k.strip()]
            keys = [k for k in want if k in set(all_keys)]
            missing = [k for k in want if k not in set(all_keys)]
            if missing:
                print(
                    f"[warn] No points for listed experiment keys (skipped): {missing[:10]}"
                    + (" ..." if len(missing) > 10 else ""),
                    file=sys.stderr,
                )
        elif args.boxplot_all_experiments:
            keys = all_keys
        else:
            keys = []

        if not keys:
            print(
                "No experiment keys for multi boxplot. Check --boxplot-experiments or your scan filters.",
                file=sys.stderr,
            )
            sys.exit(1)

        box_out = args.boxplot_output
        if box_out is None:
            if args.output:
                box_out = args.output.parent / f"{args.output.stem}_boxplots_multi{args.output.suffix}"
            else:
                print(
                    "Multi boxplot needs --boxplot-output or main -o "
                    "(file will be <stem>_boxplots_multi<suffix>).",
                    file=sys.stderr,
                )
                sys.exit(2)

        fig_multi = create_multi_experiment_checkpoint_score_boxplots(
            points,
            experiment_keys=keys,
            task_substring=args.task_substring,
            rollout_window=args.rollout_window,
            title=args.boxplot_title,
        )
        box_out = box_out.resolve()
        if box_out.suffix.lower() == ".html":
            fig_multi.write_html(str(box_out))
        else:
            fig_multi.write_image(str(box_out))
        print(f"Wrote {box_out}")
        if args.matplotlib_pdf and args.output:
            from policy_doctor.plotting.curation_scatter_mpl import (
                save_multi_experiment_boxplots_matplotlib_pdf,
            )

            main_out = args.output.resolve()
            box_pdf = main_out.parent / f"{main_out.stem}_mpl_boxplots_multi.pdf"
            save_multi_experiment_boxplots_matplotlib_pdf(
                box_pdf,
                points,
                experiment_keys=keys,
                task_substring=args.task_substring,
                rollout_window=args.rollout_window,
                title=args.boxplot_title,
            )
            print(f"Wrote {box_pdf}")


if __name__ == "__main__":
    main()
