from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def quantize(
    image: np.ndarray,
    k: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an RGB image to a palette of ``k`` colours via K-means.

    Returns ``(palette, indices)`` where ``palette`` is ``(k, 3) uint8`` and
    ``indices`` is ``(H, W) int`` with values in ``[0, k)`` pointing into the
    palette.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {image.shape}")

    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float64)

    # If the image has fewer unique colours than k, KMeans will complain about
    # n_samples < n_clusters. Deduplicate first and cap k accordingly.
    unique_pixels = np.unique(pixels, axis=0)
    effective_k = min(k, len(unique_pixels))

    kmeans = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10,
    ).fit(unique_pixels)
    labels = kmeans.predict(pixels)

    palette = np.zeros((k, 3), dtype=np.uint8)
    palette[:effective_k] = np.clip(
        np.rint(kmeans.cluster_centers_), 0, 255
    ).astype(np.uint8)
    indices = labels.reshape(h, w).astype(np.int32)
    return palette, indices
