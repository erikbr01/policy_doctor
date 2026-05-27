"""CLI: bundle an experiment into a tarball for transfer.

    python -m policy_doctor.scripts.experiment_bundle <name> [--out <path>]

Dereferences any symlinks under ``shared/`` (so the source dataset becomes a
hard copy inside the bundle) and packages the entire experiment dir as
``.tar.gz``. Resulting archive is self-contained — extract on another machine
and resume with no external dependencies beyond the source datasets.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional

from policy_doctor.experiment import Experiment


def _materialize_symlinks(root: Path) -> None:
    """Replace every symlink under root with a hard copy of its target."""
    for path in list(root.rglob("*")):
        if path.is_symlink():
            target = path.resolve()
            if not target.exists():
                raise FileNotFoundError(
                    f"Symlink {path} → {target} is broken; cannot bundle."
                )
            path.unlink()
            if target.is_dir():
                shutil.copytree(target, path)
            else:
                shutil.copy2(target, path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Experiment name to bundle.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination tarball path. Defaults to <name>.tar.gz in cwd.",
    )
    args = parser.parse_args(argv)

    try:
        exp = Experiment.load(args.name)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out_path = args.out or Path(f"{args.name}.tar.gz")
    out_path = out_path.resolve()

    # Stage the experiment in a tmpdir, dereference symlinks, then tar.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / args.name
        shutil.copytree(exp.root, staged, symlinks=True)
        _materialize_symlinks(staged)
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(staged, arcname=args.name)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Bundled '{args.name}' → {out_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
