from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pbn.edges import region_boundaries
from pbn.labels import compute_placements


# Preferred TrueType fonts, in order. ``load_default`` is used as a final
# fallback when no TrueType file is discoverable on the host system — pixel
# size isn't selectable there, so adaptive sizing degrades gracefully.
_TTF_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


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
    for path in _TTF_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=max(1, int(size)))
            except OSError:
                continue
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
    outlines, and the palette-index digit inside every region. Regions too
    small for an inscribed digit get a lead-line to a nearby roomy anchor.

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

    placements = compute_placements(indices, scale=scale)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    def _source_to_canvas(yx: tuple[int, int]) -> tuple[int, int]:
        y, x = yx
        return (y * scale + scale // 2, x * scale + scale // 2)

    for p in placements:
        digit = str(int(p["palette_index"]) + 1)
        font = _load_font(p["digit_size"])
        cy, cx = _source_to_canvas(p["digit_pos"])

        if p["leadline_from"] is not None and p["leadline_to"] is not None:
            ly, lx = _source_to_canvas(p["leadline_from"])
            ty, tx = _source_to_canvas(p["leadline_to"])
            # Shorten the line slightly at the digit end so it doesn't
            # overwrite the glyph.
            dy = ty - ly
            dx = tx - lx
            length = float(np.hypot(dy, dx))
            if length > 0:
                pad = min(length * 0.5, p["digit_size"] * 0.6)
                ly = int(round(ly + (dy / length) * pad))
                lx = int(round(lx + (dx / length) * pad))
            draw.line([(lx, ly), (tx, ty)], fill=(0, 0, 0), width=1)

        try:
            draw.text(
                (cx, cy), digit, fill=(0, 0, 0), font=font, anchor="mm"
            )
        except (TypeError, ValueError):
            # Fallback for Pillow without ``anchor`` support.
            bbox = draw.textbbox((0, 0), digit, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                (cx - tw // 2, cy - th // 2),
                digit,
                fill=(0, 0, 0),
                font=font,
            )

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
