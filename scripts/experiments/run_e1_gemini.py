"""Run E1 cluster coherence classification with a Gemini backend (VertexAI).

Reads GCP credentials from environment (GOOGLE_CLOUD_PROJECT + ADC).

Usage:
    source .env && conda run -n policy_doctor python scripts/run_e1_gemini.py \
        --clustering_dir <dir> --out_dir <out> \
        --model_name gemini-3.1-pro-preview --location global \
        --max_clusters 10 --n_example 3 --n_query 3 --n_repetitions 3
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustering_dir", required=True)
    ap.add_argument(
        "--eval_dir",
        default=(
            "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27/"
            "mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest"
        ),
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="gemini-3.1-pro-preview")
    ap.add_argument("--vertexai", action="store_true", default=True)
    ap.add_argument("--project", default=None,
                    help="GCP project (falls back to GOOGLE_CLOUD_PROJECT env var)")
    ap.add_argument("--location", default="global")
    ap.add_argument("--max_clusters", type=int, default=10)
    ap.add_argument("--n_example", type=int, default=3)
    ap.add_argument("--n_query", type=int, default=3)
    ap.add_argument("--n_repetitions", type=int, default=3)
    ap.add_argument("--composite_target_size", type=int, default=768)
    ap.add_argument("--max_frames_per_storyboard", type=int, default=4)
    ap.add_argument("--storyboard_mode", default="composite")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--global_episode_disjoint", action="store_true", default=True)
    ap.add_argument("--max_output_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    import random
    import numpy as np
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    from policy_doctor.vlm.backends.gemini import GeminiVLMBackend
    from policy_doctor.vlm.cluster_classification import run_cluster_coherence_classification

    project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    backend = GeminiVLMBackend(
        model_name=args.model_name,
        vertexai=args.vertexai,
        project=project,
        location=args.location,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[E1-Gemini] backend={backend.name} model={args.model_name}\n"
        f"  project={project} location={args.location}\n"
        f"  clustering_dir={args.clustering_dir}\n"
        f"  K_max={args.max_clusters} n_example={args.n_example} "
        f"n_query={args.n_query} n_reps={args.n_repetitions}\n"
        f"  out={out}",
        flush=True,
    )

    run_cluster_coherence_classification(
        clustering_dir=pathlib.Path(args.clustering_dir),
        eval_dir=pathlib.Path(args.eval_dir),
        backend=backend,
        n_example=args.n_example,
        n_query=args.n_query,
        n_repetitions=args.n_repetitions,
        max_frames_per_storyboard=args.max_frames_per_storyboard,
        random_seed=args.random_seed,
        step_dir=out,
        max_clusters=args.max_clusters,
        dry_run=False,
        global_episode_disjoint=args.global_episode_disjoint,
        composite_target_size=args.composite_target_size,
        storyboard_mode=args.storyboard_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
