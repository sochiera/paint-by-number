from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_bilateral

from pbn.quantize import quantize
from pbn.regions import (
    cap_fragments_per_color,
    merge_small_regions,
    merge_to_target_count,
)
from pbn.render import render_palette, render_preview, render_template
from pbn.saliency import SALIENCY_MODES, compute_saliency_weights
from pbn.segment import PRESEGMENT_CHOICES, slic_presegment


SMOOTHING_CHOICES = ("none", "gaussian", "bilateral", "meanshift")
CLEANUP_CHOICES = ("none", "majority")


@dataclass(frozen=True)
class PBNResult:
    """The full output of the paint-by-number pipeline."""

    palette: np.ndarray  # (k, 3) uint8 — only first effective_k rows used
    indices: np.ndarray  # (H, W) int32, values in [0, effective_k)
    preview: np.ndarray  # (H, W, 3) uint8
    template: np.ndarray  # (H*scale, W*scale, 3) uint8
    legend: np.ndarray  # (?, ?, 3) uint8
    effective_k: int = 0


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
    max_per_color: int | None = None,
    cleanup: str | None = "majority",
    min_delta_e: float = 7.0,
    saliency: str = "none",
    presegment: str = "none",
    slic_segments: int = 600,
    slic_compactness: float = 10.0,
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
    max_per_color : optional upper bound on the number of 4-connected
        components for each individual palette colour. Smallest fragments
        of an over-budget colour are absorbed into their longest-bordered
        neighbour of a *different* colour, so the painter's eye isn't drawn
        to scattered same-numbered specks. Runs after ``max_regions``;
        ``None`` (default) disables it.
    cleanup : label-map cleanup mode applied after quantisation and before
        region merging. ``"majority"`` (default) replaces each pixel's label
        with the most frequent label in its 3x3 neighbourhood, which
        dissolves isolated speckles without moving strong contours.
        ``"none"`` or ``None`` disables cleanup.
    min_delta_e : minimum CIE76 Lab distance between any two palette
        centroids. After the initial K-means fit, pairs closer than this
        threshold are collapsed by re-running K-means with ``k - 1`` until
        every remaining pair is at least ``min_delta_e`` apart (or only two
        centroids remain). Set to ``0`` to disable collapsing. Default ``7``.
    saliency : per-pixel weighting strategy passed as ``sample_weight`` to
        K-means. ``"none"`` (default) keeps the unweighted fit; ``"center"``
        uses a Gaussian centred on the canvas; ``"auto"`` calls
        ``cv2.saliency.StaticSaliencyFineGrained`` if available, with a
        Sobel-magnitude fallback otherwise. Pulls centroids towards the
        weighted areas so subjects in cluttered backgrounds get their own
        palette colours.
    presegment : pre-quantisation segmentation. ``"none"`` (default) feeds
        the smoothed image directly to K-means. ``"slic"`` runs SLIC and
        replaces every pixel with its superpixel's mean RGB before
        quantisation; this collapses textured regions and can drastically
        cut the eventual number of 4-connected paint regions.
    slic_segments, slic_compactness : tuning for ``presegment='slic'``;
        ignored otherwise. ``slic_segments`` is the approximate number of
        superpixels and ``slic_compactness`` is SLIC's spatial-vs-colour
        trade-off (~10 is sensible for typical photos).
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
    if saliency not in SALIENCY_MODES:
        raise ValueError(
            f"unknown saliency mode {saliency!r}; "
            f"expected one of {SALIENCY_MODES}"
        )
    if presegment not in PRESEGMENT_CHOICES:
        raise ValueError(
            f"unknown presegment mode {presegment!r}; "
            f"expected one of {PRESEGMENT_CHOICES}"
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

    # Saliency weights come from the smoothed image so edge cues survive even
    # when SLIC has flattened intra-superpixel detail in the quantize input.
    weights = compute_saliency_weights(working, mode=saliency)

    quantize_input = working
    if presegment == "slic":
        quantize_input = slic_presegment(
            working,
            n_segments=slic_segments,
            compactness=slic_compactness,
        )

    palette, indices, effective_k = quantize(
        quantize_input,
        k=k,
        random_state=random_state,
        min_delta_e=min_delta_e,
        sample_weight=weights,
    )

    if cleanup_mode == "majority":
        indices = _majority_filter(indices, size=3)

    if min_region_size > 1:
        indices = merge_small_regions(indices, min_size=min_region_size)

    if max_regions is not None:
        indices = merge_to_target_count(indices, max_regions=max_regions)

    if max_per_color is not None:
        indices = cap_fragments_per_color(
            indices, max_per_color=max_per_color
        )

    used_palette = palette[:effective_k]
    preview = render_preview(used_palette, indices)
    template = render_template(indices, used_palette, scale=template_scale)
    legend = render_palette(used_palette)

    return PBNResult(
        palette=palette,
        indices=indices,
        preview=preview,
        template=template,
        legend=legend,
        effective_k=effective_k,
    )
