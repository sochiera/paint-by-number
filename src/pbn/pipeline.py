from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_bilateral

from pbn.quantize import quantize
from pbn.regions import merge_small_regions, merge_to_target_count
from pbn.render import render_palette, render_preview, render_template


SMOOTHING_CHOICES = ("none", "gaussian", "bilateral", "meanshift")
CLEANUP_CHOICES = ("none", "majority")


@dataclass(frozen=True)
class PBNResult:
    """The full output of the paint-by-number pipeline."""

    palette: np.ndarray  # (k, 3) uint8
    indices: np.ndarray  # (H, W) int32, values in [0, k)
    preview: np.ndarray  # (H, W, 3) uint8
    template: np.ndarray  # (H*scale, W*scale, 3) uint8
    legend: np.ndarray  # (?, ?, 3) uint8


def _gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    blurred = np.empty_like(image, dtype=np.float64)
    for c in range(3):
        ndimage.gaussian_filter(
            image[..., c].astype(np.float64), sigma, output=blurred[..., c]
        )
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def _bilateral_smooth(
    image: np.ndarray, sigma_color: float, sigma_spatial: float
) -> np.ndarray:
    as_float = image.astype(np.float64) / 255.0
    filtered = denoise_bilateral(
        as_float,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=-1,
    )
    return np.clip(np.rint(filtered * 255.0), 0, 255).astype(np.uint8)


def _meanshift_smooth(image: np.ndarray, sp: float, sr: float) -> np.ndarray:
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "smooth='meanshift' requires opencv-python. "
            "Install it with 'pip install opencv-python'."
        ) from exc
    # cv2 expects BGR but mean shift is colour-symmetric enough for our
    # purposes; still, round-trip through BGR to match OpenCV's expectations.
    bgr = image[..., ::-1].copy()
    filtered = cv2.pyrMeanShiftFiltering(bgr, sp=sp, sr=sr)
    return filtered[..., ::-1].copy()


def _majority_filter(indices: np.ndarray, size: int = 3) -> np.ndarray:
    """Replace each label with the most frequent label in a ``size x size``
    neighbourhood (edge-replicated border). Ties broken by lowest label.

    Vectorised: for each unique label ``l``, compute the neighbourhood
    count of ``l`` via ``uniform_filter`` on the binary mask, then take
    the argmax over labels per pixel. Complexity ``O(k * H * W)``.
    """
    if indices.ndim != 2:
        raise ValueError(f"expected 2D indices, got shape {indices.shape}")
    labels = np.unique(indices)
    h, w = indices.shape
    # counts[l, y, x] = number of pixels with label l in the window around (y, x)
    counts = np.empty((len(labels), h, w), dtype=np.float32)
    area = float(size * size)
    for i, lbl in enumerate(labels):
        mask = (indices == lbl).astype(np.float32)
        # uniform_filter with mode='nearest' gives the mean over the
        # size x size window with edge replication; multiply by area
        # to recover the integer count (as float — exact enough for argmax).
        ndimage.uniform_filter(
            mask, size=size, output=counts[i], mode="nearest"
        )
        counts[i] *= area
    # argmax over axis 0 picks the label with the highest count; numpy's
    # argmax returns the first occurrence on ties, and `labels` is sorted
    # ascending, so ties are broken by lowest label — deterministic.
    winner = np.argmax(counts, axis=0)
    return labels[winner].astype(indices.dtype)


def _smooth(
    image: np.ndarray,
    mode: str,
    blur_sigma: float,
    bilateral_sigma_color: float,
    bilateral_sigma_spatial: float,
    meanshift_sp: float,
    meanshift_sr: float,
) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "gaussian":
        if blur_sigma <= 0:
            return image
        return _gaussian_smooth(image, blur_sigma)
    if mode == "bilateral":
        return _bilateral_smooth(
            image, bilateral_sigma_color, bilateral_sigma_spatial
        )
    if mode == "meanshift":
        return _meanshift_smooth(image, meanshift_sp, meanshift_sr)
    raise ValueError(
        f"unknown smoothing mode {mode!r}; expected one of {SMOOTHING_CHOICES}"
    )


def generate(
    image: np.ndarray,
    k: int,
    min_region_size: int = 0,
    blur_sigma: float = 0.0,
    template_scale: int = 4,
    random_state: int = 0,
    smooth: str = "gaussian",
    bilateral_sigma_color: float = 0.1,
    bilateral_sigma_spatial: float = 3.0,
    meanshift_sp: float = 10.0,
    meanshift_sr: float = 20.0,
    max_regions: int | None = None,
    cleanup: str | None = "majority",
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
        Only consulted when ``smooth == "gaussian"``.
    template_scale : nearest-neighbour upscale of the printable template so
        digits and outlines render crisply.
    random_state : seed passed to K-means for reproducible palettes.
    smooth : one of ``"none"``, ``"gaussian"``, ``"bilateral"``,
        ``"meanshift"``. Selects the pre-quantisation smoothing filter.
        ``"bilateral"`` and ``"meanshift"`` preserve edges while flattening
        textured regions. ``"none"`` skips smoothing even if ``blur_sigma``
        is positive.
    bilateral_sigma_color, bilateral_sigma_spatial : parameters for the
        bilateral filter; only consulted when ``smooth == "bilateral"``.
    meanshift_sp, meanshift_sr : spatial and colour radii for mean-shift
        filtering; only consulted when ``smooth == "meanshift"``. Requires
        ``opencv-python`` to be installed.
    max_regions : optional upper bound on the number of 4-connected regions
        in the output. When set, the smallest regions are iteratively merged
        into the adjacent region with the longest shared border until the
        total drops to this value. ``None`` (default) disables the stage.
    cleanup : label-map cleanup mode applied after quantisation and before
        region merging. ``"majority"`` (default) replaces each pixel's label
        with the most frequent label in its 3x3 neighbourhood, which
        dissolves isolated speckles without moving strong contours.
        ``"none"`` or ``None`` disables cleanup.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB image, got {image.shape}")
    if smooth not in SMOOTHING_CHOICES:
        raise ValueError(
            f"unknown smoothing mode {smooth!r}; "
            f"expected one of {SMOOTHING_CHOICES}"
        )
    cleanup_mode = "none" if cleanup is None else cleanup
    if cleanup_mode not in CLEANUP_CHOICES:
        raise ValueError(
            f"unknown cleanup mode {cleanup!r}; "
            f"expected one of {CLEANUP_CHOICES}"
        )

    working = _smooth(
        image,
        smooth,
        blur_sigma=blur_sigma,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_spatial=bilateral_sigma_spatial,
        meanshift_sp=meanshift_sp,
        meanshift_sr=meanshift_sr,
    )

    palette, indices = quantize(working, k=k, random_state=random_state)

    if cleanup_mode == "majority":
        indices = _majority_filter(indices, size=3)

    if min_region_size > 1:
        indices = merge_small_regions(indices, min_size=min_region_size)

    if max_regions is not None:
        indices = merge_to_target_count(indices, max_regions=max_regions)

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
