"""
Policy comparison entry point — powered by Hydra.

Reads a comparison config (groups of eval dirs) and produces:
  - violin.png    — Beta posterior distributions with CLD letters
  - bar.png       — empirical success rates with Wilson CI error bars
  - breakdown.png — per-run strip plot showing variance across seeds/checkpoints
  - summary.json  — CLD letters, rates, n_episodes, per-run detail
  - details.csv   — one row per run
  - spec.yaml     — resolved config (reproducibility record)

Usage:
  python -m policy_doctor.scripts.compare_policies \\
    comparison=mimicgen_apr26_demos60_seed1

  # override output dir or title on the fly
  python -m policy_doctor.scripts.compare_policies \\
    comparison=mimicgen_apr26_demos60_seed1 \\
    comparison.output_dir=data/outputs/comparisons/my_run \\
    comparison.title="My custom title"
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from policy_doctor.paths import CONFIGS_DIR, REPO_ROOT


@hydra.main(
    config_path=str(CONFIGS_DIR),
    config_name="compare_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    comp = cfg.get("comparison")
    if comp is None:
        raise ValueError(
            "No comparison config loaded. Select one with comparison=<name>.\n"
            "Available: " + ", ".join(
                p.stem for p in (CONFIGS_DIR / "comparison").glob("*.yaml")
            )
        )

    repo_root = Path(cfg.get("repo_root") or REPO_ROOT)
    eval_output_dir = repo_root / (comp.eval_output_dir or "data/outputs/eval_save_episodes")

    output_dir = Path(
        comp.get("output_dir") or "data/outputs/comparisons/unnamed"
    )
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare_policies] output → {output_dir}")

    # Build PolicyGroupSpec from config
    from policy_doctor.analysis.policy_comparison import (
        PolicyGroupSpec,
        print_comparison_table,
        run_comparison,
    )

    group_specs = []
    for g in comp.groups:
        dirs = [eval_output_dir / name for name in g.run_names]
        group_specs.append(PolicyGroupSpec(label=g.label, dirs=dirs))

    if len(group_specs) < 2:
        raise ValueError("Need at least 2 groups to compare.")

    import numpy as np
    rng_seed = int(comp.get("rng_seed") or 42)
    rng = np.random.default_rng(rng_seed)

    top_k = comp.get("top_k_checkpoints")
    result = run_comparison(
        group_specs,
        global_confidence_level=float(comp.get("global_confidence_level") or 0.95),
        method=str(comp.get("method") or "barnard_exact"),
        max_sample_size=comp.get("max_sample_size_per_policy") or None,
        top_k_checkpoints=int(top_k) if top_k is not None else 5,
        shuffle=bool(comp.get("shuffle") if comp.get("shuffle") is not None else True),
        rng=rng,
        verbose=True,
    )
    print_comparison_table(result)

    # Plots
    import matplotlib.pyplot as plt
    from policy_doctor.plotting.policy_comparison import (
        create_policy_comparison_bar,
        create_policy_comparison_breakdown,
        create_policy_comparison_violin,
    )

    title = comp.get("title") or None
    plot_rng = np.random.default_rng(rng_seed)

    fig = create_policy_comparison_violin(
        result.group_labels, result.success_arrays, result.cld_letters,
        rng=plot_rng, show_empirical_means=True, title=title,
    )
    fig.savefig(output_dir / "violin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  violin    → {output_dir / 'violin.png'}")

    fig = create_policy_comparison_bar(
        result.group_labels, result.success_rates, result.n_episodes,
        result.cld_letters,
        confidence=float(comp.get("global_confidence_level") or 0.95),
        title=title,
    )
    fig.savefig(output_dir / "bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  bar       → {output_dir / 'bar.png'}")

    breakdown_data = [
        [(r.name, r.success_rate, r.n_episodes) for r in runs]
        for runs in result.per_group_runs
    ]
    fig = create_policy_comparison_breakdown(
        result.group_labels, breakdown_data, result.cld_letters, title=title,
    )
    fig.savefig(output_dir / "breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  breakdown → {output_dir / 'breakdown.png'}")

    # Summary JSON
    summary = {
        "method": result.method,
        "global_confidence_level": result.global_confidence_level,
        "max_sample_size": result.max_sample_size,
        "shuffle": result.shuffle,
        "groups": [
            {
                "label": lbl,
                "cld_letter": cld,
                "success_rate": round(rate, 4),
                "n_episodes": n,
                "runs": [
                    {"path": str(r.path), "n_episodes": r.n_episodes,
                     "success_rate": round(r.success_rate, 4)}
                    for r in runs
                ],
            }
            for lbl, cld, rate, n, runs in zip(
                result.group_labels, result.cld_letters,
                result.success_rates, result.n_episodes,
                result.per_group_runs,
            )
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  summary   → {output_dir / 'summary.json'}")

    # Details CSV
    rows = []
    for lbl, cld, runs in zip(result.group_labels, result.cld_letters, result.per_group_runs):
        for r in runs:
            rows.append({
                "group": lbl, "cld_letter": cld, "run": r.name,
                "path": str(r.path), "n_episodes": r.n_episodes,
                "success_rate": round(r.success_rate, 4),
            })
    if rows:
        with open(output_dir / "details.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  details   → {output_dir / 'details.csv'}")

    # Resolved spec for reproducibility
    (output_dir / "spec.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(f"  spec      → {output_dir / 'spec.yaml'}")


if __name__ == "__main__":
    main()
