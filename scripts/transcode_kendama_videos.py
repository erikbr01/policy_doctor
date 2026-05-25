"""Transcode kendama rollout videos to web-friendly H.264 + faststart.

The raw recordings are MPEG-4 Visual (codec=mpeg4) at 1280x720 / ~4.6 Mbps,
which most browsers refuse to play in HTML5 <video>. This script re-encodes
each exterior.mp4 to:
  * codec     : H.264 (baseline profile, level 3.0) — universally supported
  * resolution: 640x360 (study videos are watched at thumbnail-ish sizes anyway)
  * bitrate   : CRF 26 capped at ~900 kbps — typically 4-7 MB per ~45 s clip
  * audio     : stripped (rollouts have no useful audio)
  * faststart : moov atom at the front so playback can start before the full
                file lands

Output naming + index.json layout matches the existing
`scripts/cluster_kendama_rollouts.py` convention so the demo app and
study app pick the new videos up transparently.

Usage (policy_doctor env, parallel ffmpeg workers):
    python scripts/transcode_kendama_videos.py \\
        --rollouts /mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final \\
        --out_dir  /tmp/study_mp4s/kendama_may22 \\
        --workers  8
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _episode_records(rollouts_dir: Path) -> list[dict]:
    """Return (ep_dir, success, n_steps) for episodes with both an mp4 and meta."""
    out = []
    for ep_dir in sorted(rollouts_dir.iterdir()):
        mp4 = ep_dir / "exterior.mp4"
        meta_path = ep_dir / "meta.json"
        if not (mp4.exists() and meta_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        s = meta.get("success")
        success = bool(s) and float(s) > 0
        out.append(dict(
            ep_dir=ep_dir,
            success=success,
            n_steps=int(meta.get("n_steps", 0)),
        ))
    return out


def _detect_fps(mp4_path: Path) -> float:
    """Read avg_frame_rate from ffprobe (e.g. '15/1' → 15.0)."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path),
    ], text=True).strip()
    num, denom = out.split("/")
    return float(num) / float(denom) if float(denom) else 0.0


def _transcode_one(src: Path, dst: Path) -> tuple[Path, int]:
    """Run ffmpeg once. Returns (dst, output filesize-in-bytes)."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", "scale=640:360",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "26",
        "-maxrate", "900k",
        "-bufsize", "1800k",
        "-an",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst, dst.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--rollouts",
        default="/mnt/ssdB/erik/rollouts/rollouts_kendama_baseline_250_may22_final",
    )
    ap.add_argument(
        "--out_dir",
        default="/tmp/study_mp4s/kendama_may22",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--keep_existing",
        action="store_true",
        help="Skip re-encode for any dst file that already exists.",
    )
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")

    rollouts_dir = Path(args.rollouts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = _episode_records(rollouts_dir)
    print(f"{len(eps)} valid episodes in {rollouts_dir.name} "
          f"({sum(e['success'] for e in eps)} successes)")

    # Detect source fps once. The transcoder preserves the source rate (no
    # -r flag in the ffmpeg cmd), so every output has the same fps.
    src_fps = _detect_fps(eps[0]["ep_dir"] / "exterior.mp4")
    print(f"source fps: {src_fps:.3f}")

    # Plan output paths and matching index entries.
    jobs: list[tuple[Path, Path]] = []
    index_eps: list[dict] = []
    for idx, ep in enumerate(eps):
        suffix = "succ" if ep["success"] else "fail"
        dst = out_dir / f"ep{idx:04d}_{suffix}.mp4"
        src = ep["ep_dir"] / "exterior.mp4"
        index_eps.append(dict(
            index=idx,
            path=str(dst),
            frame_count=ep["n_steps"],
            fps=src_fps,
            success=ep["success"],
        ))
        if args.keep_existing and dst.exists():
            continue
        if dst.exists():
            dst.unlink()
        jobs.append((src, dst))

    print(f"transcoding {len(jobs)} clips with {args.workers} workers …")
    total_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_transcode_one, s, d): d for s, d in jobs}
        for i, fut in enumerate(cf.as_completed(futs)):
            dst, size = fut.result()
            total_bytes += size
            print(f"  [{i+1:>3}/{len(jobs)}] {dst.name}  ({size/1024:.0f} KB)")

    # Rewrite index.json from scratch — old entries may reference may19 paths.
    (out_dir / "index.json").write_text(json.dumps({"episodes": index_eps}, indent=2))
    if total_bytes:
        print(f"\nWrote {len(jobs)} clips, total {total_bytes/1e6:.1f} MB")
    print(f"index.json → {out_dir / 'index.json'} ({len(index_eps)} episodes)")


if __name__ == "__main__":
    main()
