from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pbn.quantize import quantize
from pbn.regions import merge_small_regions
from pbn.render import render_palette, render_preview, render_template


@dataclass(frozen=True)
class PBNResult:
    """The full output of the paint-by-number pipeline."""

    palette: np.ndarray  # (k, 3) uint8
    indices: np.ndarray  # (H, W) int32, values in [0, k)
    preview: np.ndarray  # (H, W, 3) uint8
    template: np.ndarray  # (H*scale, W*scale, 3) uint8
    legend: np.ndarray  # (?, ?, 3) uint8


def generate(
    image: np.ndarray,
    k: int,
    min_region_size: int = 0,
    blur_sigma: float = 0.0,
    template_scale: int = 4,
    random_state: int = 0,
) -> PBNResult:
    """Turn an RGB image into a paint-by-number template bundle.

    Parameters
    ----------
    image : (H, W, 3) uint8 RGB input.
    k : number of palette colours.
    min_region_size : regions smaller than this (in pixels) are merged into
        the adjacent region with which they share the longest boundary.
        Use ``0`` or ``1`` to skip merging.
    blur_sigma : Gaussian blur applied before quantisation; 0 disables it.
    template_scale : nearest-neighbour upscale of the printable template so
        digits and outlines render crisply.
    random_state : seed passed to K-means for reproducible palettes.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB image, got {image.shape}")

    working = image
    if blur_sigma > 0:
        blurred = np.empty_like(image, dtype=np.float64)
        for c in range(3):
            ndimage.gaussian_filter(
                image[..., c].astype(np.float64), blur_sigma, output=blurred[..., c]
            )
        working = np.clip(np.rint(blurred), 0, 255).astype(np.uint8)

    palette, indices = quantize(working, k=k, random_state=random_state)

    if min_region_size > 1:
        indices = merge_small_regions(indices, min_size=min_region_size)

    preview = render_preview(palette, indices)
    template = render_template(indices, palette, scale=template_scale)
    legend = render_palette(palette)

    return PBNResult(
        palette=palette,
        indices=indices,
        preview=preview,
        template=template,
        legend=legend,
    )
