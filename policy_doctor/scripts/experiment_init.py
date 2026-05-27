"""CLI: create a new self-contained experiment directory.

    python -m policy_doctor.scripts.experiment_init <name> \\
        [--baseline-from <other_experiment>]

Creates ``$POLICY_DOCTOR_DATA/experiments/<name>/`` with the standard skeleton
(``manifest.yaml``, ``config/``, ``shared/``, ``artifacts/``, ``logs/``) and
optionally hard-copies a baseline checkpoint from another experiment.

Default data root is ``<repo>/data/`` — override via the ``POLICY_DOCTOR_DATA``
environment variable.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from policy_doctor.experiment import Experiment


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Experiment name (becomes the dir name).")
    parser.add_argument(
        "--baseline-from",
        default=None,
        help="Name of an existing experiment whose shared/baseline_ckpt should be hard-copied in.",
    )
    args = parser.parse_args(argv)

    try:
        exp = Experiment.create(args.name, baseline_from=args.baseline_from)
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"Created experiment at: {exp.root}")
    if args.baseline_from:
        print(f"Baseline checkpoint copied from experiment '{args.baseline_from}'.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
