from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage


# 4-connectivity cross
_STRUCTURE_4 = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=bool,
)

_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def label_regions(indices: np.ndarray) -> np.ndarray:
    """Label 4-connected regions of equal palette index.

    Returns an ``(H, W) int32`` array with labels numbered contiguously
    from 0 to ``N - 1``.
    """
    if indices.ndim != 2:
        raise ValueError(f"expected 2D indices, got shape {indices.shape}")

    out = np.zeros_like(indices, dtype=np.int32)
    next_label = 0
    for value in np.unique(indices):
        mask = indices == value
        labelled, count = ndimage.label(mask, structure=_STRUCTURE_4)
        # Shift labels so they do not collide with previously assigned ones.
        shifted = np.where(labelled > 0, labelled + next_label - 1, 0)
        out[mask] = shifted[mask]
        next_label += count
    return out


def merge_small_regions(indices: np.ndarray, min_size: int) -> np.ndarray:
    """Absorb each 4-connected region smaller than ``min_size`` into the
    neighbouring region with the longest shared boundary.

    Iterates until every surviving region has at least ``min_size`` pixels,
    or no more progress can be made (e.g. an image made entirely of a single
    colour that is itself below the threshold).
    """
    if min_size <= 1:
        return indices.copy()

    result = indices.copy()
    h, w = result.shape
    while True:
        labels = label_regions(result)
        sizes = np.bincount(labels.ravel())
        small_labels = np.where(sizes < min_size)[0]
        if len(small_labels) == 0:
            break

        progress = False
        for lbl in small_labels:
            # The label map may be stale after an earlier merge in this pass;
            # re-check size each time.
            mask = labels == lbl
            if not mask.any():
                continue

            neighbour_counts: Counter[int] = Counter()
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                for dy, dx in _NEIGHBOUR_OFFSETS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] != lbl:
                        neighbour_counts[int(result[ny, nx])] += 1

            if not neighbour_counts:
                # Fully enclosed by the image border and contains the whole
                # image — nothing to merge into.
                continue

            best_colour = max(
                neighbour_counts,
                key=lambda c: (neighbour_counts[c], -c),
            )
            result[mask] = best_colour
            progress = True

        if not progress:
            break
    return result
