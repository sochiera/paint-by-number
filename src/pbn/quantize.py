from __future__ import annotations

import numpy as np
from skimage.color import deltaE_cie76, lab2rgb, rgb2lab
from sklearn.cluster import KMeans


def _fit_kmeans(
    lab_pixels: np.ndarray,
    n_clusters: int,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit(lab_pixels, sample_weight=sample_weight)
    # ``predict`` is unweighted (nearest-centroid by Euclidean distance), which
    # is what we want for the per-pixel index map — every pixel still gets the
    # closest palette colour, even where its sample weight is small.
    labels = kmeans.predict(lab_pixels)
    return kmeans.cluster_centers_, labels


def _min_pairwise_delta_e(centroids_lab: np.ndarray) -> tuple[float, int, int]:
    """Return ``(min_delta, i, j)`` over distinct pairs of Lab centroids.

    ``i < j`` is guaranteed; ``(0, 0, 0)`` when fewer than 2 centroids.
    """
    n = len(centroids_lab)
    if n < 2:
        return float("inf"), 0, 0
    best = float("inf")
    best_i, best_j = 0, 1
    for i in range(n):
        deltas = deltaE_cie76(
            centroids_lab[i][None, :], centroids_lab[i + 1 :]
        )
        if deltas.size == 0:
            continue
        local = int(np.argmin(deltas))
        local_d = float(deltas[local])
        if local_d < best:
            best = local_d
            best_i = i
            best_j = i + 1 + local
    return best, best_i, best_j


def _collapse_close_lab_centroids(
    lab_pixels: np.ndarray,
    centroids_lab: np.ndarray,
    labels: np.ndarray,
    min_delta_e: float,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively drop the perceptually-closest centroid pair by re-running
    K-means with ``k - 1`` clusters. ``sample_weight`` is forwarded so the
    re-fits stay consistent with the initial weighted fit. Stops when all
    remaining pairs have ``delta_e >= min_delta_e`` or only two centroids
    remain.
    """
    if min_delta_e <= 0:
        return centroids_lab, labels

    while len(centroids_lab) > 2:
        min_d, _, _ = _min_pairwise_delta_e(centroids_lab)
        if min_d >= min_delta_e:
            break
        new_k = len(centroids_lab) - 1
        centroids_lab, labels = _fit_kmeans(
            lab_pixels,
            n_clusters=new_k,
            random_state=random_state,
            sample_weight=sample_weight,
        )

    return centroids_lab, labels


def quantize(
    image: np.ndarray,
    k: int,
    random_state: int = 0,
    min_delta_e: float = 7.0,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reduce an RGB image to a palette of at most ``k`` colours via K-means
    in CIELab, collapsing centroid pairs closer than ``min_delta_e`` (CIE76).

    Parameters
    ----------
    sample_weight : optional ``(H, W)`` or flat ``(H*W,)`` array of
        non-negative per-pixel weights. Pulls centroids towards heavily
        weighted pixels (e.g. saliency maps); per-pixel index assignment
        stays unweighted so every pixel still maps to its closest centroid.

    Returns ``(palette, indices, effective_k)`` where:
      * ``palette`` is ``(k, 3) uint8`` — only the first ``effective_k`` rows
        hold real centroids; the remainder is zero-padded.
      * ``indices`` is ``(H, W) int32`` with values in ``[0, effective_k)``.
      * ``effective_k`` is the number of palette slots actually used.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got {image.shape}")
    if min_delta_e < 0:
        raise ValueError(f"min_delta_e must be >= 0, got {min_delta_e}")

    h, w, _ = image.shape
    lab_image = rgb2lab(image)
    lab_pixels = lab_image.reshape(-1, 3)

    weight_flat: np.ndarray | None = None
    if sample_weight is not None:
        weight_flat = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weight_flat.shape[0] != h * w:
            raise ValueError(
                f"sample_weight has {weight_flat.shape[0]} entries; expected "
                f"{h * w} (image is {h}x{w})"
            )
        if (weight_flat < 0).any():
            raise ValueError("sample_weight must be non-negative")

    unique_rgb = np.unique(image.reshape(-1, 3), axis=0)
    initial_k = min(k, len(unique_rgb))

    centroids_lab, labels = _fit_kmeans(
        lab_pixels,
        n_clusters=initial_k,
        random_state=random_state,
        sample_weight=weight_flat,
    )

    centroids_lab, labels = _collapse_close_lab_centroids(
        lab_pixels,
        centroids_lab,
        labels,
        min_delta_e=min_delta_e,
        random_state=random_state,
        sample_weight=weight_flat,
    )

    # Reindex labels into a contiguous ``[0, effective_k)`` range so the
    # palette prefix aligns with the indices seen in the output.
    used = np.unique(labels)
    remap = np.full(len(centroids_lab), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    labels = remap[labels]
    centroids_lab = centroids_lab[used]
    effective_k = len(centroids_lab)

    centroids_rgb = lab2rgb(centroids_lab.reshape(1, -1, 3)).reshape(-1, 3)

    palette = np.zeros((k, 3), dtype=np.uint8)
    palette[:effective_k] = np.clip(
        np.rint(centroids_rgb * 255.0), 0, 255
    ).astype(np.uint8)
    indices = labels.reshape(h, w).astype(np.int32)
    return palette, indices, effective_k
