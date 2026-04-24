from __future__ import annotations

import numpy as np
from scipy import ndimage


def label_positions(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """For every connected region in ``labels`` return a ``(x, y, label)``
    tuple where ``(x, y)`` is the pixel inside that region furthest from
    any region boundary — i.e. the safest spot to print the number.
    """
    if labels.ndim != 2:
        raise ValueError(f"expected 2D labels, got shape {labels.shape}")

    results: list[tuple[int, int, int]] = []
    for lbl in np.unique(labels):
        mask = labels == lbl
        # Pad with a zero border so the image edge counts as boundary too:
        # we want the label anchor to sit away from both inner seams and the
        # outer frame.
        padded = np.pad(mask, 1, constant_values=False)
        dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
        flat_idx = int(np.argmax(dist))
        y, x = np.unravel_index(flat_idx, dist.shape)
        results.append((int(x), int(y), int(lbl)))
    return results
