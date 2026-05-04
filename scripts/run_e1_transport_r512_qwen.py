"""Smoke run of Experiment E1 on transport_mh r512 / seed0 with Qwen3-VL-8B.

Bypasses the Hydra pipeline because the clustering at /tmp/transport_mh_seed0_r512_clustering
references a task_config that no longer exists as a YAML. We supply the eval_dir
explicitly and call run_cluster_coherence_classification directly.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Ensure the repo's policy_doctor (this worktree) wins over the pip-installed
# editable copy that may live in a sibling worktree.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustering_dir", default="/tmp/transport_mh_seed0_r512_clustering")
    ap.add_argument(
        "--eval_dir",
        default=(
            "/mnt/ssdB/erik/cupid_data/outputs/eval_save_episodes/mar27/"
            "mar27_train_diffusion_unet_lowdim_transport_mh_0_r512x512/latest"
        ),
    )
    ap.add_argument("--out_dir", default="experiments/e1_transport_r512_seed0_qwen3vl8b")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_clusters", type=int, default=5)
    ap.add_argument("--n_example", type=int, default=2)
    ap.add_argument("--n_query", type=int, default=2)
    ap.add_argument("--n_repetitions", type=int, default=1)
    ap.add_argument("--max_frames_per_storyboard", type=int, default=4)
    ap.add_argument("--random_seed", type=int, default=42)
    # Cap on per-image pixel budget after the storyboard is built. 1.05M px ≈ 1024×1024.
    ap.add_argument("--image_max_pixels", type=int, default=1024 * 1024)
    ap.add_argument(
        "--global_episode_disjoint",
        action="store_true",
        help="Block cross-cluster episode contamination by preferring queries "
             "whose episode appears in NO cluster's example pool.",
    )
    ap.add_argument(
        "--view_window_extension",
        type=int,
        default=0,
        help="Widen the visual frame window symmetrically by this many timesteps "
             "on each side without changing the cluster window. Tests whether "
             "longer visual context recovers more accuracy at fixed clustering.",
    )
    ap.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load weights via bitsandbytes NF4. Required for Qwen3-VL-32B on a "
             "single 24GB GPU (~16GB quantized weights vs 64GB at bf16).",
    )
    ap.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load weights via bitsandbytes 8-bit. Roughly 32GB for Qwen3-VL-32B "
             "— needs device_map=auto across two GPUs.",
    )
    ap.add_argument(
        "--device_map",
        default=None,
        help="HF/accelerate device_map. Use 'auto' to spread layers across all "
             "visible GPUs (required for 8-bit Qwen3-VL-32B; optional for 4-bit "
             "if a single GPU is too tight).",
    )
    ap.add_argument(
        "--include_action_text",
        action="store_true",
        help="Append per-timestep executed action vectors as text after each "
             "slice's storyboard. Adds tokens but gives the VLM access to the "
             "policy's actual action sequence.",
    )
    ap.add_argument(
        "--include_state_text",
        action="store_true",
        help="Append per-timestep current-frame observation vectors as text "
             "after each slice's storyboard. Adds tokens; complements the "
             "image with the same numeric state the policy receives.",
    )
    args = ap.parse_args()

    import policy_doctor as _pd
    print("policy_doctor pkg:", _pd.__file__, flush=True)

    # Seed every layer of randomness *before* backend init: numpy (sample plan,
    # frame sampling, label maps), torch + cuda (any future do_sample=True),
    # and Python random (third-party utilities). This makes the run
    # reproducible from --random_seed alone.
    import os, random
    import numpy as np
    import torch
    seed = int(args.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    from policy_doctor.vlm.backends.qwen2_vl import build_qwen2_vl_backend
    from policy_doctor.vlm.cluster_classification import (
        run_cluster_coherence_classification,
    )

    backend_params = {
        "model_id": args.model_id,
        "device": args.device,
        "torch_dtype": "bfloat16",
        "max_new_tokens": 64,
        "image_max_pixels": args.image_max_pixels,
        # default image input mode (separate image tokens, not video)
        "qwen_frame_input": "images",
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
    }
    if args.device_map is not None:
        backend_params["device_map"] = args.device_map
    backend = build_qwen2_vl_backend(backend_params)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[E1] backend={backend.name} model={args.model_id} device={args.device}\n"
        f"     clustering_dir={args.clustering_dir}\n"
        f"     eval_dir={args.eval_dir}\n"
        f"     K_max={args.max_clusters} n_example={args.n_example} "
        f"n_query={args.n_query} n_reps={args.n_repetitions}\n"
        f"     out={out}",
        flush=True,
    )

    summary = run_cluster_coherence_classification(
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
        view_window_extension=args.view_window_extension,
        include_action_text=args.include_action_text,
        include_state_text=args.include_state_text,
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))
    metrics_path = pathlib.Path(summary.get("metrics_path") or out / "metrics.json")
    if metrics_path.exists():
        print("\n=== metrics.json ===")
        print(metrics_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
