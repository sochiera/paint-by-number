import numpy as np
import pytest

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


def test_quantize_deterministic_with_random_state():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)

    p1, i1 = quantize(img, k=4, random_state=123)
    p2, i2 = quantize(img, k=4, random_state=123)

    assert np.array_equal(p1, p2)
    assert np.array_equal(i1, i2)
