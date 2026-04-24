from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an (H, W, 3) uint8 RGB array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def save_image(array: np.ndarray, path: str | Path) -> None:
    """Save an (H, W, 3) uint8 RGB array to a PNG/JPEG file."""
    if array.dtype != np.uint8:
        raise ValueError(f"expected uint8 array, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) shape, got {array.shape}")
    Image.fromarray(array).save(Path(path))
