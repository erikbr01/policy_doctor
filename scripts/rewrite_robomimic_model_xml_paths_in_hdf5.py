#!/usr/bin/env python3
"""Rewrite absolute mesh/texture paths inside robomimic/MimicGen HDF5 ``model_file`` XML.

If ``generate_dataset`` ran while Python resolved ``robosuite`` to a bogus prefix (e.g.
``.../third_party/mimicgen/mimicgen/lib/python3.8/site-packages/robosuite``), MuJoCo's
saved XML in ``data/demo_*/attrs:model_file`` will embed those paths. Playback then looks
for STLs in the wrong place.

This script updates every ``file="..."`` attribute in that XML to match **this**
environment's ``robosuite`` / ``mimicgen`` / ``robosuite_task_zoo`` install roots (same
idea as ``SingleArmEnv_MG.edit_model_xml``), and writes the HDF5 in place.

Run inside ``conda activate mimicgen`` so ``import robosuite`` is the stack you want
paths rewritten to. Examples::

    python scripts/rewrite_robomimic_model_xml_paths_in_hdf5.py --dry-run demo.hdf5
    python scripts/rewrite_robomimic_model_xml_paths_in_hdf5.py --backup demo.hdf5 demo_failed.hdf5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np


def _attr_to_str(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, (np.bytes_, np.str_)):
        return str(val)
    if isinstance(val, str):
        return val
    return str(val)


def _path_remap_packages() -> List[Tuple[str, List[str]]]:
    """(path_segment_marker, root_dir_split) in apply order."""
    import robosuite

    packs: List[Tuple[str, List[str]]] = [
        ("robosuite", os.path.split(robosuite.__file__)[0].split("/")),
    ]
    try:
        import mimicgen

        mg = os.path.split(mimicgen.__file__)[0].split("/")
        packs.append(("mimicgen", mg))
        packs.append(("mimicgen_envs", mg))
    except ImportError:
        pass
    try:
        import robosuite_task_zoo

        packs.append(
            ("robosuite_task_zoo", os.path.split(robosuite_task_zoo.__file__)[0].split("/"))
        )
    except ImportError:
        pass
    return packs


def remap_mesh_file_path(path: str, packs: List[Tuple[str, List[str]]]) -> Tuple[str, bool]:
    """If ``path`` contains a segment like ``.../robosuite/...``, re-root after last marker."""
    for marker, root_parts in packs:
        parts = path.split("/")
        idxs = [i for i, p in enumerate(parts) if p == marker]
        if not idxs:
            continue
        ind = max(idxs)
        new_path = "/".join(root_parts + parts[ind + 1 :])
        if new_path != path:
            return new_path, True
        return path, False
    return path, False


def rewrite_model_xml(xml_str: str, packs: List[Tuple[str, List[str]]]) -> Tuple[str, int]:
    """Return (new_xml, num_file_attributes_changed)."""
    root = ET.fromstring(xml_str)
    n = 0
    for elem in root.iter():
        fp = elem.get("file")
        if not fp:
            continue
        new_fp, changed = remap_mesh_file_path(fp, packs)
        if changed:
            elem.set("file", new_fp)
            n += 1
    # Py3.8+: encoding="unicode" returns str
    out = ET.tostring(root, encoding="unicode")
    return out, n


def iter_groups_with_model_file(h5: h5py.File) -> List[h5py.Group]:
    found: List[h5py.Group] = []

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Group) and "model_file" in obj.attrs:
            found.append(obj)

    h5.visititems(visitor)
    return found


def process_file(path: Path, *, dry_run: bool, backup: bool) -> Tuple[int, int]:
    """Return (groups_seen, total_file_attrs_rewritten)."""
    if not path.is_file():
        print(f"skip (not a file): {path}", file=sys.stderr)
        return 0, 0

    if backup and not dry_run:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"backup: {bak}")

    packs = _path_remap_packages()
    mode = "r" if dry_run else "r+"
    groups_seen = 0
    total_rewrites = 0

    try:
        with h5py.File(path, mode) as f:
            groups = iter_groups_with_model_file(f)
            for grp in groups:
                groups_seen += 1
                raw = grp.attrs["model_file"]
                xml_str = _attr_to_str(raw)
                try:
                    new_xml, n = rewrite_model_xml(xml_str, packs)
                except ET.ParseError as e:
                    print(f"{path} {grp.name}: invalid model_file XML: {e}", file=sys.stderr)
                    continue
                total_rewrites += n
                if n and not dry_run:
                    del grp.attrs["model_file"]
                    grp.attrs["model_file"] = new_xml
                elif n and dry_run:
                    print(f"  {path} :: {grp.name} :: would rewrite {n} file= attributes")
    except OSError as e:
        print(f"cannot open {path}: {e}", file=sys.stderr)
        return 0, 0

    if not dry_run and groups_seen:
        if total_rewrites:
            print(f"{path}: touched {groups_seen} episode(s), rewrote {total_rewrites} file= path(s)")
        else:
            print(f"{path}: {groups_seen} episode(s), paths already match this env")
    elif dry_run and groups_seen and total_rewrites == 0:
        print(f"{path}: {groups_seen} episode(s), no path changes needed")

    return groups_seen, total_rewrites


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hdf5", nargs="+", type=Path, help="robomimic / MimicGen dataset file(s)")
    p.add_argument("--dry-run", action="store_true", help="report changes only")
    p.add_argument(
        "--backup",
        action="store_true",
        help="copy each file to <name>.bak before modifying (no backup in dry-run)",
    )
    args = p.parse_args()

    tot_g = tot_r = 0
    for path in args.hdf5:
        g, r = process_file(path.resolve(), dry_run=args.dry_run, backup=args.backup)
        tot_g += g
        tot_r += r

    if args.dry_run and tot_r:
        print(f"dry-run: would rewrite {tot_r} file= attribute(s) across {tot_g} episode group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
