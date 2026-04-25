import numpy as np
import pytest

from pbn.segment import PRESEGMENT_CHOICES, slic_presegment


def _noisy_two_block_image(size: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.int16)
    img[:, : size // 2] = (200, 60, 60)
    img[:, size // 2 :] = (60, 60, 200)
    img += rng.integers(-25, 26, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_slic_presegment_returns_same_shape_and_dtype():
    img = _noisy_two_block_image()
    out = slic_presegment(img, n_segments=20, compactness=10.0)

    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_slic_presegment_collapses_intra_segment_noise():
    img = _noisy_two_block_image()
    unique_in = len(np.unique(img.reshape(-1, 3), axis=0))
    out = slic_presegment(img, n_segments=20, compactness=10.0)
    unique_out = len(np.unique(out.reshape(-1, 3), axis=0))

    assert unique_out < unique_in / 5, (
        f"presegment should collapse noise; "
        f"unique RGB went {unique_in} -> {unique_out}"
    )


def test_slic_presegment_preserves_dominant_block_colors():
    img = _noisy_two_block_image(size=80)
    out = slic_presegment(img, n_segments=80, compactness=10.0)

    # Sample well away from the boundary so mixed-segment means don't
    # dilute the test.
    left_mean = out[:, :15].astype(np.int32).mean(axis=(0, 1))
    right_mean = out[:, 65:].astype(np.int32).mean(axis=(0, 1))
    assert left_mean[0] > left_mean[2] + 80
    assert right_mean[2] > right_mean[0] + 80


def test_slic_presegment_rejects_bad_n_segments():
    img = _noisy_two_block_image()
    with pytest.raises(ValueError):
        slic_presegment(img, n_segments=0, compactness=10.0)


def test_slic_presegment_rejects_bad_image_shape():
    bad = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        slic_presegment(bad, n_segments=4, compactness=10.0)


def test_choices_lists_known_modes():
    assert set(PRESEGMENT_CHOICES) == {"none", "slic"}
