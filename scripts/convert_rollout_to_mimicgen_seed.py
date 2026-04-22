"""Convert a policy rollout HDF5 to robomimic seed format for MimicGen.

This script reads a single episode from an ``eval_save_episodes`` rollout HDF5
and writes it as a robomimic-style seed HDF5 that can be passed to
``run_mimicgen_generate.py`` (or ``mimicgen.scripts.prepare_src_dataset``
directly).

Usage::

    python scripts/convert_rollout_to_mimicgen_seed.py \\
        --input  data/outputs/eval_save_episodes/.../rollouts.hdf5 \\
        --output /tmp/seed_from_rollout.hdf5 \\
        --demo_key demo_0

The output can then be fed into MimicGen generation::

    conda run -n mimicgen python scripts/run_mimicgen_generate.py \\
        --seed_hdf5 /tmp/seed_from_rollout.hdf5 \\
        --output_dir /tmp/gen_output \\
        --task_name square \\
        --num_trials 50

Note on MuJoCo compatibility: rollouts collected via the **cupid** env use
MuJoCo 3.x; NVlabs MimicGen source data targets MuJoCo 2.3.x.  If
``prepare_src_dataset`` fails with XML parser errors, re-record the seed demo
with the mimicgen env or strip the ``model_file`` attribute (``--no_model_file``
flag).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a policy rollout episode to robomimic seed HDF5 for MimicGen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, metavar="HDF5",
                        help="Path to the rollout HDF5 (eval_save_episodes output).")
    parser.add_argument("--output", required=True, metavar="HDF5",
                        help="Path for the output seed HDF5.")
    parser.add_argument("--demo_key", default="demo_0",
                        help="Episode key to read (default: demo_0).")
    parser.add_argument("--no_model_file", action="store_true",
                        help="Strip the model_file attribute (useful when MuJoCo versions differ).")
    parser.add_argument("--list_demos", action="store_true",
                        help="List available demo keys in the input HDF5 and exit.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    # Allow running from repo root without installing policy_doctor
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root / "policy_doctor") not in sys.path:
        sys.path.insert(0, str(repo_root))

    import h5py

    # List mode
    if args.list_demos:
        with h5py.File(input_path, "r") as f:
            keys = sorted(k for k in f.get("data", {}).keys() if k.startswith("demo_"))
        print(f"Available demo keys in {input_path}:")
        for k in keys:
            print(f"  {k}")
        return 0

    from policy_doctor.mimicgen.materializer import RobomimicSeedMaterializer
    from policy_doctor.mimicgen.seed_trajectory import MimicGenSeedTrajectory

    print(f"Loading rollout: {input_path} [{args.demo_key}]")
    traj = MimicGenSeedTrajectory.from_rollout_hdf5(input_path, demo_key=args.demo_key)

    if args.no_model_file:
        from dataclasses import replace
        traj = replace(traj, model_file=None)

    output_path = Path(args.output)
    print(f"Writing seed HDF5: {output_path}")
    mat = RobomimicSeedMaterializer()
    mat.write_source_dataset(
        states=traj.states,
        actions=traj.actions,
        env_meta=traj.env_meta,
        output_path=output_path,
        model_file=traj.model_file,
    )

    print(
        f"\nDone."
        f"\n  source   : {traj.source.value}"
        f"\n  timesteps: {traj.states.shape[0]}"
        f"\n  state_dim: {traj.states.shape[1]}"
        f"\n  action_dim: {traj.actions.shape[1]}"
        f"\n  model_file: {'<present>' if traj.model_file else 'None'}"
        f"\n  env_name : {traj.env_meta.get('env_name', '?')}"
        f"\n  output   : {output_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
