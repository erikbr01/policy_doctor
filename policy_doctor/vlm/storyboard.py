"""Composite multiple frames into a single storyboard image for VLM prompts.

Reduces token cost: K×n_example×n_frames images → K×n_example composite images.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
from PIL import Image


def make_storyboard(
    images: Sequence[Image.Image],
    *,
    target_size: Tuple[int, int] = (512, 512),
    background: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Arrange *images* into a grid composite of *target_size*.

    If len(images) == 1, returns a resized single image.
    Otherwise tiles into a near-square grid (rows × cols where cols >= rows).
    Empty cells are filled with *background*.
    """
    n = len(images)
    if n == 0:
        raise ValueError("make_storyboard requires at least one image")

    if n == 1:
        return images[0].convert("RGB").resize(target_size, Image.LANCZOS)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = target_size[0] // cols
    cell_h = target_size[1] // rows

    canvas = Image.new("RGB", target_size, background)
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        thumb = img.convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        canvas.paste(thumb, (col * cell_w, row * cell_h))

    return canvas
