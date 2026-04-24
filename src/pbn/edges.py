from __future__ import annotations

import numpy as np


def region_boundaries(indices: np.ndarray) -> np.ndarray:
    """Return a boolean mask of pixels that are adjacent to a pixel of a
    different palette index (4-connectivity).

    The outer image border is NOT treated as a boundary — only transitions
    that happen inside the image count.
    """
    if indices.ndim != 2:
        raise ValueError(f"expected 2D indices, got shape {indices.shape}")

    edges = np.zeros(indices.shape, dtype=bool)
    # Compare each pixel with its four neighbours; mark both sides of any
    # mismatch so outlines are 2 pixels wide.
    diff_down = indices[:-1, :] != indices[1:, :]
    edges[:-1, :] |= diff_down
    edges[1:, :] |= diff_down

    diff_right = indices[:, :-1] != indices[:, 1:]
    edges[:, :-1] |= diff_right
    edges[:, 1:] |= diff_right
    return edges
