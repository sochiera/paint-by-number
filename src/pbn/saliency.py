"""Per-pixel weight maps used to bias K-means towards salient image areas.

The pipeline calls :func:`compute_saliency_weights` and forwards the result as
``sample_weight`` to :func:`pbn.quantize.quantize`. Heavy weights pull
centroids towards the corresponding pixels, so the resulting palette spends
more of its budget on whatever the saliency map deems important.

Weights are clipped to ``[floor, +inf)`` and normalised so their mean equals
``1.0``. The clip prevents background pixels from being silently zeroed (which
can starve a centroid completely and break the K-means re-fit).
"""

from __future__ import annotations

import numpy as np

SALIENCY_MODES = ("none", "center", "auto")

# Default lower bound on the normalised weight. Stops large flat backgrounds
# from being wholly ignored — they still get a centroid, just a smaller share
# of the budget.
_WEIGHT_FLOOR = 0.2


def _center_weights(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sigma = max(h, w) / 4.0
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)).astype(
        np.float32
    )


def _auto_weights(image: np.ndarray) -> np.ndarray:
    """OpenCV's StaticSaliencyFineGrained, with a Sobel-magnitude fallback
    for environments without ``opencv-contrib-python``. Both produce a
    ``(H, W) float32`` map in roughly ``[0, 1]``."""
    try:
        import cv2  # noqa: PLC0415

        if hasattr(cv2, "saliency"):
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            bgr = image[..., ::-1].copy()
            ok, sal_map = saliency.computeSaliency(bgr)
            if ok:
                return sal_map.astype(np.float32)
    except ImportError:
        pass

    # Fallback: Sobel gradient magnitude on the L channel (Lab). High
    # gradient = boundaries / textured detail, which we treat as salient.
    from skimage.color import rgb2lab  # noqa: PLC0415
    from skimage.filters import sobel  # noqa: PLC0415

    lab = rgb2lab(image)
    grad = sobel(lab[..., 0])
    return grad.astype(np.float32)


def compute_saliency_weights(image: np.ndarray, mode: str) -> np.ndarray | None:
    """Return a ``(H, W) float32`` weight map (mean ≈ 1) for the given mode,
    or ``None`` when the mode disables weighting.

    ``mode`` choices:
      * ``"none"`` — return ``None``; quantize stays unweighted.
      * ``"center"`` — Gaussian centred on the canvas; cheapest, no extra deps.
      * ``"auto"`` — ``cv2.saliency.StaticSaliencyFineGrained`` if available,
        otherwise a Sobel-magnitude fallback on the Lab L channel.
    """
    if mode == "none":
        return None
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {image.shape}")
    if mode == "center":
        raw = _center_weights(image.shape[:2])
    elif mode == "auto":
        raw = _auto_weights(image)
    else:
        raise ValueError(
            f"unknown saliency mode {mode!r}; expected one of {SALIENCY_MODES}"
        )

    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != image.shape[:2]:
        raise ValueError(
            f"saliency map shape {raw.shape} != image shape {image.shape[:2]}"
        )

    # Rescale to [0, 1] then floor + renormalise to mean 1.
    rmin, rmax = float(raw.min()), float(raw.max())
    if rmax > rmin:
        raw = (raw - rmin) / (rmax - rmin)
    else:
        raw = np.ones_like(raw)
    raw = np.clip(raw, _WEIGHT_FLOOR, 1.0)
    raw = raw / raw.mean()
    return raw.astype(np.float32)
