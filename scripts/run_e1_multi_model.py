"""Multi-model E1 cluster coherence classification.

All specified models receive identical queries and label maps so their
predictions are directly comparable. Inter-model agreement is captured
as a first-class metric in metrics.json.

Model specs are passed as comma-separated triples: ``name:backend:params``
where ``params`` is a JSON string.  The ``--models`` flag is designed to be
readable while still flexible.  See examples below.

Usage (smoke test with mock backend):
    conda run -n policy_doctor python scripts/run_e1_multi_model.py \\
        --clustering_dir <dir> --out_dir /tmp/e1_multi_test \\
        --models "mock_a:mock:{}" "mock_b:mock:{}" \\
        --max_clusters 2 --n_example 2 --n_query 1 --n_repetitions 1

Usage (Qwen + Gemini):
    source .env && \\
    conda run -n policy_doctor_torch2 python scripts/run_e1_multi_model.py \\
        --clustering_dir <dir> --out_dir <out> \\
        --models \\
            'qwen3_vl:qwen3_vl:{"model_id":"Qwen/Qwen3-VL-8B-Instruct","device":"cuda:0","load_in_4bit":true}' \\
            'gemini25flash:gemini:{"model_name":"gemini-2.5-flash","vertexai":true,"location":"us-central1"}' \\
        --max_clusters 10 --n_example 3 --n_query 5 --n_repetitions 3
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import List, Tuple

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _build_backend(name: str, backend_type: str, params: dict):
    """Instantiate a VLMBackend from type string + params dict."""
    bt = backend_type.lower().strip()
    if bt == "mock":
        from policy_doctor.vlm.backends.mock import MockVLMBackend
        return MockVLMBackend(**(params or {}))
    if bt in ("qwen2_vl", "qwen3_vl", "qwen"):
        from policy_doctor.vlm.backends.qwen2_vl import build_qwen2_vl_backend
        return build_qwen2_vl_backend(params or {})
    if bt == "gemini":
        from policy_doctor.vlm.backends.gemini import build_gemini_backend
        return build_gemini_backend(params or {})
    if bt == "claude":
        from policy_doctor.vlm.backends.claude import build_claude_backend
        return build_claude_backend(params or {})
    raise ValueError(
        f"Unknown backend type {backend_type!r}. "
        "Choose from: mock, qwen3_vl, gemini, claude."
    )


def parse_model_spec(spec: str) -> Tuple[str, str, dict]:
    """Parse ``name:backend_type:json_params`` → (name, backend_type, params)."""
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise ValueError(
            f"Model spec must be 'name:backend_type' or 'name:backend_type:{{json}}', got {spec!r}"
        )
    name = parts[0].strip()
    backend_type = parts[1].strip()
    params = json.loads(parts[2]) if len(parts) == 3 and parts[2].strip() else {}
    return name, backend_type, params


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clustering_dir", required=True)
    ap.add_argument(
        "--eval_dir",
        default=(
            "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27/"
            "mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest"
        ),
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME:BACKEND[:JSON_PARAMS]",
        help=(
            "One or more model specs, each as 'name:backend_type:{json_params}'. "
            "backend_type: mock, qwen3_vl, gemini, claude."
        ),
    )
    ap.add_argument("--max_clusters", type=int, default=None)
    ap.add_argument("--n_example", type=int, default=3)
    ap.add_argument("--n_query", type=int, default=5)
    ap.add_argument("--n_repetitions", type=int, default=3)
    ap.add_argument("--max_frames_per_storyboard", type=int, default=4)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--global_episode_disjoint", action="store_true")
    ap.add_argument("--composite_target_size", type=int, default=768)
    ap.add_argument("--storyboard_mode", default="composite")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    from policy_doctor.vlm.cluster_classification import run_cluster_coherence_classification

    # Load .env if present
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    # Build backends
    backends: List[Tuple[str, object]] = []
    for spec in args.models:
        name, backend_type, params = parse_model_spec(spec)
        print(f"  Loading backend: {name!r} ({backend_type}) params={params}", flush=True)
        bk = _build_backend(name, backend_type, params)
        backends.append((name, bk))
        print(f"    OK: {bk}", flush=True)

    print(
        f"\n[E1-multi] {len(backends)} model(s): {[n for n, _ in backends]}\n"
        f"  clustering_dir={args.clustering_dir}\n"
        f"  eval_dir={args.eval_dir}\n"
        f"  K_max={args.max_clusters} n_example={args.n_example} "
        f"n_query={args.n_query} n_reps={args.n_repetitions}\n"
        f"  out={args.out_dir}",
        flush=True,
    )

    summary = run_cluster_coherence_classification(
        clustering_dir=pathlib.Path(args.clustering_dir),
        eval_dir=pathlib.Path(args.eval_dir),
        backends=backends,
        n_example=args.n_example,
        n_query=args.n_query,
        n_repetitions=args.n_repetitions,
        max_frames_per_storyboard=args.max_frames_per_storyboard,
        random_seed=args.random_seed,
        step_dir=pathlib.Path(args.out_dir),
        max_clusters=args.max_clusters,
        dry_run=args.dry_run,
        global_episode_disjoint=args.global_episode_disjoint,
        composite_target_size=args.composite_target_size,
        storyboard_mode=args.storyboard_mode,
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))

    metrics_path = pathlib.Path(summary.get("metrics_path") or args.out_dir + "/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print("\n=== metrics.json (top-level) ===")
        top_keys = ["top1_accuracy", "chance_level", "binomial_test_pvalue",
                    "inter_model_agreement", "rep_agreement_rate_mean", "model_names"]
        for k in top_keys:
            if k in metrics:
                print(f"  {k}: {metrics[k]}")
        if "per_model" in metrics:
            print("\n=== per-model accuracy ===")
            for mn, m in metrics["per_model"].items():
                acc = m.get("top1_accuracy", "n/a")
                pval = m.get("binomial_test_pvalue", "n/a")
                print(f"  {mn}: {acc:.3f}  p={pval}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
