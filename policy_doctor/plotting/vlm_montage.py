"""PIL montage of RGB frames for VLM preview (no Streamlit)."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import List, Sequence

import numpy as np
from PIL import Image


def create_frame_montage(
    images: Sequence[Image.Image],
    *,
    max_cols: int = 4,
    thumb_size: tuple[int, int] = (160, 120),
    pad: int = 4,
    bg_color: tuple[int, int, int] = (40, 40, 40),
) -> Image.Image:
    """Tile RGB images into one PIL image (left-to-right, then wrap)."""
    if not images:
        raise ValueError("images must be non-empty")
    thumbs: List[Image.Image] = []
    for im in images:
        rgb = im.convert("RGB")
        thumbs.append(rgb.resize(thumb_size, Image.Resampling.BILINEAR))

    n = len(thumbs)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    tw, th = thumb_size
    w = cols * (tw + pad) + pad
    h = rows * (th + pad) + pad
    canvas = Image.new("RGB", (w, h), bg_color)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + pad)
        canvas.paste(t, (x, y))
    return canvas


def pil_to_uint8_rgb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def create_scrollable_frame_strip_html(
    images: Sequence[Image.Image],
    *,
    thumb_height_px: int = 160,
    gap_px: int = 8,
    container_max_width: str = "100%",
) -> str:
    """HTML snippet: flex row with horizontal overflow (for ``st.markdown(..., unsafe_allow_html=True)``)."""
    if not images:
        raise ValueError("images must be non-empty")
    parts: List[str] = [
        f'<div style="max-width:{container_max_width}; overflow-x:auto; overflow-y:hidden; '
        f'-webkit-overflow-scrolling:touch; padding:4px 0;">',
        f'<div style="display:flex; flex-direction:row; flex-wrap:nowrap; gap:{gap_px}px; '
        f'align-items:flex-end; width:max-content;">',
    ]
    for im in images:
        buf = BytesIO()
        im.convert("RGB").save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        parts.append(
            f'<img src="data:image/png;base64,{b64}" alt="" '
            f'style="height:{thumb_height_px}px; width:auto; flex-shrink:0; display:block; '
            f'border-radius:4px; object-fit:contain;" />'
        )
    parts.append("</div></div>")
    return "\n".join(parts)
