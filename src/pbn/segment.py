"""Optional pre-segmentation step that flattens an image into superpixels
before quantisation.

After SLIC labels every pixel with a superpixel id, we replace each pixel's
RGB with its superpixel's mean RGB. K-means then sees a much smaller set of
distinct colours (one per superpixel rather than one per pixel) and the
clustering naturally collapses textured regions, cutting down the eventual
number of 4-connected paint regions.
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import slic

PRESEGMENT_CHOICES = ("none", "slic")


def slic_presegment(
    image: np.ndarray,
    n_segments: int = 600,
    compactness: float = 10.0,
) -> np.ndarray:
    """Replace each pixel with its SLIC superpixel's mean RGB.

    Parameters
    ----------
    image : ``(H, W, 3) uint8`` RGB.
    n_segments : approximate number of superpixels SLIC should produce. Higher
        values preserve more detail; lower values flatten more aggressively.
    compactness : SLIC's spatial-vs-colour trade-off. ~10 is a sensible
        default for typical photos.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {image.shape}")
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    if compactness <= 0:
        raise ValueError(f"compactness must be > 0, got {compactness}")

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
        convert2lab=True,
    )

    # Vectorised per-segment mean: weight each segment by its pixel count and
    # sum colours via bincount, then write back.
    n = int(segments.max()) + 1
    flat_seg = segments.ravel()
    flat_img = image.reshape(-1, 3).astype(np.float64)
    counts = np.bincount(flat_seg, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid divide-by-zero for absent labels
    means = np.empty((n, 3), dtype=np.float64)
    for c in range(3):
        means[:, c] = (
            np.bincount(flat_seg, weights=flat_img[:, c], minlength=n)
            / counts
        )
    out = means[flat_seg].reshape(image.shape)
    return np.clip(np.rint(out), 0, 255).astype(image.dtype)
