from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pbn.edges import region_boundaries
from pbn.labels import label_positions
from pbn.regions import label_regions


def render_preview(palette: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Paint each pixel with the palette colour named by ``indices``.

    Returns an ``(H, W, 3) uint8`` RGB array — i.e. what the finished
    paint-by-number should look like when every region has been coloured in.
    """
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must be (k, 3), got {palette.shape}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D, got shape {indices.shape}")
    if indices.min() < 0 or indices.max() >= len(palette):
        raise IndexError(
            f"indices out of palette range [0, {len(palette)}): "
            f"min={indices.min()}, max={indices.max()}"
        )
    return palette[indices].astype(np.uint8, copy=False)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def render_template(
    indices: np.ndarray,
    palette: np.ndarray,
    scale: int = 1,
) -> np.ndarray:
    """Render the painter-facing template: white canvas, black region
    outlines, and the palette-index digit inside every region.

    Parameters
    ----------
    indices : (H, W) integer array — palette index per pixel.
    palette : (k, 3) uint8 array — only its length is used here.
    scale : int — nearest-neighbour upscale factor so digits are legible.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    # Upscale by nearest-neighbour so the outline/digit rendering happens at
    # the painter's resolution.
    big = np.repeat(np.repeat(indices, scale, axis=0), scale, axis=1)

    canvas = np.full((*big.shape, 3), 255, dtype=np.uint8)
    canvas[region_boundaries(big)] = 0

    # Label regions on the original resolution so tiny single-pixel regions
    # share one anchor even after upscaling.
    labels = label_regions(indices)
    positions = label_positions(labels)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font_size = max(8, scale * 2)
    font = _load_font(font_size)

    for x, y, _region_lbl in positions:
        digit = str(int(indices[y, x]) + 1)
        cx = x * scale + scale // 2
        cy = y * scale + scale // 2
        try:
            draw.text((cx, cy), digit, fill=(0, 0, 0), font=font, anchor="mm")
        except (TypeError, ValueError):
            # Fallback for Pillow without ``anchor`` support.
            bbox = draw.textbbox((0, 0), digit, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - th // 2), digit, fill=(0, 0, 0), font=font)

    return np.asarray(img, dtype=np.uint8)


def render_palette(palette: np.ndarray, swatch_size: int = 40) -> np.ndarray:
    """Render a vertical legend of ``palette`` as ``number | swatch | RGB``."""
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must be (k, 3), got {palette.shape}")
    if swatch_size < 4:
        raise ValueError(f"swatch_size must be >= 4, got {swatch_size}")

    k = len(palette)
    row_h = swatch_size + 4
    font = _load_font(max(10, swatch_size // 2))

    # Measure the widest "RGB(r, g, b)" label so nothing gets clipped.
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_label_w = 0
    for r, g, b in palette:
        bbox = measure.textbbox((0, 0), f"RGB({int(r)}, {int(g)}, {int(b)})", font=font)
        max_label_w = max(max_label_w, bbox[2] - bbox[0])

    text_left = swatch_size * 2 + 8
    width = text_left + max_label_w + swatch_size // 2
    height = row_h * k
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    # Paint swatches as raw array slices so they round-trip exactly, which
    # the tests assert on.
    for i, colour in enumerate(palette):
        top = i * row_h + 2
        left = swatch_size  # leaves room for the digit to the left
        canvas[top : top + swatch_size, left : left + swatch_size] = colour

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    for i, colour in enumerate(palette):
        top = i * row_h + 2
        text_y = top + swatch_size // 2
        draw.text(
            (swatch_size // 4, text_y),
            str(i + 1),
            fill=(0, 0, 0),
            font=font,
            anchor="lm",
        )
        r, g, b = (int(c) for c in colour)
        draw.text(
            (text_left, text_y),
            f"RGB({r}, {g}, {b})",
            fill=(0, 0, 0),
            font=font,
            anchor="lm",
        )

    return np.asarray(img, dtype=np.uint8)
