"""Pure PIL frame-annotation helper.

Extracted from the legacy ``influence_visualizer.plotting.frames`` module so
``policy_doctor`` streamlit code can keep rendering label-overlaid frames after
the iv package is removed. No Streamlit / plotly dependencies — returns a PIL
Image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_annotated_frame(
    img: np.ndarray,
    label: str,
    font_size: int = 12,
) -> Image.Image:
    """Create an annotated PIL Image with a text label overlay in the top-left.

    Args:
        img: RGB numpy array ``(H, W, 3)`` or ``(H, W)`` grayscale.
        label: Text to overlay on the image.
        font_size: Font size for the overlay text.

    Returns:
        PIL Image with annotation.
    """
    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except Exception:
            font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    padding = 3
    x, y = padding, padding

    # Draw background rectangle for visibility
    draw.rectangle(
        [x - 2, y - 2, x + text_width + 4, y + text_height + 4],
        fill=(0, 0, 0, 180),
    )

    draw.text((x, y), label, fill=(255, 255, 255), font=font)

    return pil_img
