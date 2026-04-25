"""Resolve user-friendly print parameters into the pipeline's pixel-space
defaults.

Painters care about millimetres on paper, not source pixels. ``--print-size``
and ``--dpi`` let them say "I want an A4 print at 300 DPI" and we work back to
the integer ``--scale`` (so the upscaled template fits the page) and the
``--min-region`` threshold (so no painted region is smaller than 4 mm²).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRINT_SIZES = ("A4", "A3", "Letter")

# Page sizes in millimetres (short side, long side). Orientation is chosen at
# resolve time based on the input aspect ratio.
_PAGE_DIMENSIONS_MM: dict[str, tuple[float, float]] = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "Letter": (215.9, 279.4),
}

# Smallest paint region we still consider paintable. 4 mm² is a thumbnail
# sized speck; anything smaller usually merges visually with the line.
_MIN_REGION_AREA_MM2 = 4.0


@dataclass(frozen=True)
class PrintResolution:
    """Pixel-space defaults derived from a print-size + dpi + image shape."""

    scale: int
    min_region_size: int


def _mm_to_px(mm: float, dpi: int) -> float:
    return mm * dpi / 25.4


def resolve_print_params(
    page: str,
    dpi: int,
    image_h: int,
    image_w: int,
) -> PrintResolution:
    """Return (scale, min_region_size) for printing ``image_h × image_w``
    onto ``page`` at ``dpi``. Page orientation matches the image's aspect.

    ``scale`` is the largest integer that still fits the image on the page;
    ``min_region_size`` is the smallest source-pixel region that maps to
    at least 4 mm² on the printed output.
    """
    if page not in _PAGE_DIMENSIONS_MM:
        raise ValueError(
            f"unknown print size {page!r}; expected one of {PRINT_SIZES}"
        )
    if dpi < 1:
        raise ValueError(f"dpi must be >= 1, got {dpi}")
    if image_h < 1 or image_w < 1:
        raise ValueError(
            f"image dimensions must be positive, got {image_h}x{image_w}"
        )

    short_mm, long_mm = _PAGE_DIMENSIONS_MM[page]
    if image_w >= image_h:
        page_w_mm, page_h_mm = long_mm, short_mm
    else:
        page_w_mm, page_h_mm = short_mm, long_mm

    page_w_px = _mm_to_px(page_w_mm, dpi)
    page_h_px = _mm_to_px(page_h_mm, dpi)

    raw_scale = min(page_w_px / image_w, page_h_px / image_h)
    scale = max(1, int(math.floor(raw_scale)))

    px_per_mm = dpi / 25.4
    region_px_per_mm2_output = px_per_mm**2
    # Source pixels needed to land >= _MIN_REGION_AREA_MM2 on the output:
    # source_px * scale^2 >= mm2 * (dpi/25.4)^2
    min_region_size = max(
        1,
        int(
            math.ceil(_MIN_REGION_AREA_MM2 * region_px_per_mm2_output / scale**2)
        ),
    )
    return PrintResolution(scale=scale, min_region_size=min_region_size)
