import numpy as np
import pytest

from pbn.saliency import SALIENCY_MODES, compute_saliency_weights


def _gradient_image(size: int = 32) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, size, dtype=np.uint8)[:, None]
    return img


def test_none_returns_none():
    img = _gradient_image()
    assert compute_saliency_weights(img, "none") is None


def test_center_mode_peaks_in_middle():
    img = _gradient_image(size=64)
    w = compute_saliency_weights(img, "center")

    assert w is not None
    assert w.shape == img.shape[:2]
    assert w.dtype == np.float32

    h, ww = w.shape
    cy, cx = h // 2, ww // 2
    # Centre weight must beat all four corners.
    corners = np.array(
        [w[0, 0], w[0, ww - 1], w[h - 1, 0], w[h - 1, ww - 1]]
    )
    assert w[cy, cx] > corners.max()


def test_weights_have_unit_mean_and_floor():
    img = _gradient_image(size=64)
    for mode in ("center", "auto"):
        w = compute_saliency_weights(img, mode)
        assert w is not None
        assert w.min() > 0, f"{mode}: weights must respect a positive floor"
        assert abs(float(w.mean()) - 1.0) < 1e-3, (
            f"{mode}: mean weight {w.mean():.4f} should be 1.0"
        )


def test_unknown_mode_raises():
    img = _gradient_image()
    with pytest.raises(ValueError):
        compute_saliency_weights(img, "bogus")


def test_modes_constant_lists_known_values():
    assert set(SALIENCY_MODES) == {"none", "center", "auto"}
