from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans


def quantize(
    image: np.ndarray,
    k: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an RGB image to a palette of ``k`` colours via K-means in CIELab.

    Returns ``(palette, indices)`` where ``palette`` is ``(k, 3) uint8`` and
    ``indices`` is ``(H, W) int`` with values in ``[0, k)`` pointing into the
    palette.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {image.shape}")

    h, w, _ = image.shape
    lab_image = rgb2lab(image)
    lab_pixels = lab_image.reshape(-1, 3)

    unique_rgb = np.unique(image.reshape(-1, 3), axis=0)
    effective_k = min(k, len(unique_rgb))

    kmeans = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10,
    ).fit(lab_pixels)
    labels = kmeans.predict(lab_pixels)

    centroids_lab = kmeans.cluster_centers_
    centroids_rgb = lab2rgb(centroids_lab.reshape(1, -1, 3)).reshape(-1, 3)

    palette = np.zeros((k, 3), dtype=np.uint8)
    palette[:effective_k] = np.clip(
        np.rint(centroids_rgb * 255.0), 0, 255
    ).astype(np.uint8)
    indices = labels.reshape(h, w).astype(np.int32)
    return palette, indices
