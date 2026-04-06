"""Save matplotlib debug figures for VLM slice annotation (exact model I/O). No Streamlit."""

from __future__ import annotations

import pathlib
import textwrap
from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image


def save_slice_annotation_debug_png(
    path,
    *,
    images: Sequence[Image.Image],
    system_prompt: Optional[str],
    user_prompt: str,
    model_label: str,
    meta: Dict[str, Any],
    text_wrap: int = 96,
) -> None:
    """Write one PNG: frames exactly passed to ``describe_slice``, plus prompts and model text.

    Parameters
    ----------
    path
        Output ``.png`` path (parent dirs created).
    meta
        Should include at least slice_index, rollout_idx, window_start, window_end, cluster_id, backend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    fig_w = min(3.2 * max(n, 1), 22.0)
    fig_h = 5.5 + min(0.12 * max(len(user_prompt), len(model_label)) / 80, 6.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)

    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[1.0, 0.55 + min(0.02 * (len(user_prompt) + len(model_label)) / 200, 1.2)],
        hspace=0.22,
    )

    if n == 0:
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.text(0.5, 0.5, "(no images)", ha="center", va="center")
        ax0.axis("off")
    else:
        inner = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=gs[0], wspace=0.08)
        for j, im in enumerate(images):
            ax = fig.add_subplot(inner[0, j])
            arr = np.asarray(im.convert("RGB"))
            ax.imshow(arr)
            ax.set_title(f"to model {j + 1}/{n}", fontsize=9)
            ax.axis("off")

    ax_txt = fig.add_subplot(gs[1, 0])
    ax_txt.axis("off")

    def _blk(title: str, body: str) -> str:
        body = body.strip() if body else ""
        if not body:
            return f"=== {title} ===\n(empty)"
        return f"=== {title} ===\n{textwrap.fill(body, text_wrap)}"

    header = (
        f"slice_index={meta.get('slice_index')}  rollout_idx={meta.get('rollout_idx')}  "
        f"window=[{meta.get('window_start')}, {meta.get('window_end')})  "
        f"cluster_id={meta.get('cluster_id')}  backend={meta.get('backend')}"
    )
    parts = [header]
    if system_prompt and str(system_prompt).strip():
        parts.append(_blk("system prompt (to model)", str(system_prompt)))
    parts.append(_blk("user prompt (to model)", user_prompt))
    parts.append(_blk("model output", model_label))

    ax_txt.text(
        0.01,
        0.99,
        "\n\n".join(parts),
        transform=ax_txt.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        wrap=False,
    )

    fig.suptitle("VLM slice debug — inputs match describe_slice()", fontsize=10, y=1.02)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
