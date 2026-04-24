import numpy as np
import pytest
from skimage.color import deltaE_cie76, rgb2lab

from pbn.quantize import quantize


def _two_color_image():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :2] = (255, 0, 0)
    img[:, 2:] = (0, 255, 0)
    return img


def test_quantize_returns_expected_shapes():
    img = _two_color_image()
    palette, indices = quantize(img, k=2, random_state=0)

    assert palette.dtype == np.uint8
    assert palette.shape == (2, 3)
    assert indices.dtype.kind in ("i", "u")
    assert indices.shape == (4, 4)


def test_quantize_recovers_two_colors_exactly():
    img = _two_color_image()
    palette, indices = quantize(img, k=2, random_state=0)

    palette_set = {tuple(c) for c in palette}
    assert palette_set == {(255, 0, 0), (0, 255, 0)}

    reconstructed = palette[indices]
    assert np.array_equal(reconstructed, img)


def test_quantize_indices_in_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    k = 5
    palette, indices = quantize(img, k=k, random_state=0)

    assert palette.shape == (k, 3)
    assert indices.min() >= 0
    assert indices.max() < k


def test_quantize_k_larger_than_unique_colors_collapses():
    img = _two_color_image()
    palette, indices = quantize(img, k=8, random_state=0)

    unique_used = np.unique(indices)
    # only 2 distinct colors exist, so no more than 2 indices should actually be used
    assert len(unique_used) == 2


def test_quantize_rejects_bad_k():
    img = _two_color_image()
    with pytest.raises(ValueError):
        quantize(img, k=0)
    with pytest.raises(ValueError):
        quantize(img, k=-1)


def _dominant_grey_image_with_accents(size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((size, size, 3), 128, dtype=np.int16)
    noise = rng.integers(-12, 13, (size, size, 3), dtype=np.int16)
    img = img + noise

    gradient = np.linspace(60, 200, size, dtype=np.int16)
    img[:, :, 0] = np.clip(img[:, :, 0] + (gradient[None, :] - 128) // 3, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] + (gradient[None, :] - 128) // 3, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] + (gradient[None, :] - 128) // 3, 0, 255)

    accents = [
        ((5, 20, 5, 20), (220, 30, 30)),
        ((5, 20, 30, 45), (30, 180, 40)),
        ((5, 20, 55, 70), (40, 60, 220)),
        ((30, 45, 5, 20), (230, 210, 40)),
        ((30, 45, 30, 45), (200, 100, 30)),
        ((30, 45, 55, 70), (120, 30, 180)),
    ]
    for (r0, r1, c0, c1), colour in accents:
        img[r0:r1, c0:c1] = colour

    return np.clip(img, 0, 255).astype(np.uint8)


def test_lab_palette_is_perceptually_diverse():
    img = _dominant_grey_image_with_accents()
    k = 12
    palette, _ = quantize(img, k=k, random_state=0)

    lab = rgb2lab(palette.reshape(1, -1, 3)).reshape(-1, 3)

    diverse_count = 0
    for i in range(k):
        others = np.delete(lab, i, axis=0)
        min_delta = deltaE_cie76(lab[i][None, :], others).min()
        if min_delta >= 15:
            diverse_count += 1

    assert diverse_count >= 6, (
        f"expected >= 6 perceptually distinct colours, got {diverse_count}"
    )


def test_quantize_deterministic_with_random_state():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)

    p1, i1 = quantize(img, k=4, random_state=123)
    p2, i2 = quantize(img, k=4, random_state=123)

    assert np.array_equal(p1, p2)
    assert np.array_equal(i1, i2)
