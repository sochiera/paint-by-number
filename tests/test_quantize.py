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
    palette, indices, effective_k = quantize(img, k=2, random_state=0)

    assert palette.dtype == np.uint8
    assert palette.shape == (2, 3)
    assert indices.dtype.kind in ("i", "u")
    assert indices.shape == (4, 4)
    assert isinstance(effective_k, int)
    assert 1 <= effective_k <= 2


def test_quantize_recovers_two_colors_exactly():
    img = _two_color_image()
    palette, indices, effective_k = quantize(img, k=2, random_state=0)

    palette_set = {tuple(c) for c in palette[:effective_k]}
    assert palette_set == {(255, 0, 0), (0, 255, 0)}

    reconstructed = palette[indices]
    assert np.array_equal(reconstructed, img)


def test_quantize_indices_in_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    k = 5
    palette, indices, effective_k = quantize(
        img, k=k, random_state=0, min_delta_e=0.0
    )

    assert palette.shape == (k, 3)
    assert indices.min() >= 0
    assert indices.max() < effective_k
    assert effective_k <= k


def test_quantize_k_larger_than_unique_colors_collapses():
    img = _two_color_image()
    palette, indices, effective_k = quantize(img, k=8, random_state=0)

    unique_used = np.unique(indices)
    # only 2 distinct colors exist, so no more than 2 indices should actually be used
    assert len(unique_used) == 2
    assert effective_k == 2


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
    palette, _, effective_k = quantize(
        img, k=k, random_state=0, min_delta_e=0.0
    )

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

    p1, i1, k1 = quantize(img, k=4, random_state=123, min_delta_e=0.0)
    p2, i2, k2 = quantize(img, k=4, random_state=123, min_delta_e=0.0)

    assert np.array_equal(p1, p2)
    assert np.array_equal(i1, i2)
    assert k1 == k2


def _near_duplicate_grey_image(size: int = 96) -> np.ndarray:
    """Build an image whose dominant tones are a ladder of near-duplicate
    greys plus a couple of strongly-separated anchor colours. Default
    K-means with ``k=12`` will place several centroids on the grey ladder
    (pairs ΔE ≈ 2-6), so the collapse loop must merge them; two anchors
    remain perceptually distinct so ``effective_k >= 2`` after collapsing.
    """
    rng = np.random.default_rng(0)
    img = np.zeros((size, size, 3), dtype=np.int16)
    # Top 3/4: a ladder of near-duplicate mid-greys (the "noise" K-means
    # will waste many centroids on).
    grey_band = (3 * size) // 4
    greys = [118, 122, 126, 130, 134, 138, 142, 146]
    for i, g in enumerate(greys):
        r0 = (i * grey_band) // len(greys)
        r1 = ((i + 1) * grey_band) // len(greys)
        img[r0:r1, :, :] = g
    # Bottom quarter: two high-contrast anchors (near-black and near-white)
    # so even aggressive collapsing leaves at least two well-separated
    # centroids in the palette.
    anchor_band = size - grey_band
    split = grey_band + anchor_band // 2
    img[grey_band:split, :, :] = 20
    img[split:, :, :] = 235
    noise = rng.integers(-2, 3, (size, size, 3), dtype=np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def test_palette_respects_min_delta_e():
    img = _near_duplicate_grey_image()
    threshold = 10.0
    palette, _, effective_k = quantize(
        img, k=12, random_state=0, min_delta_e=threshold
    )

    used = palette[:effective_k]
    lab = rgb2lab(used.reshape(1, -1, 3)).reshape(-1, 3)
    for i in range(effective_k):
        for j in range(i + 1, effective_k):
            delta = float(
                deltaE_cie76(lab[i][None, :], lab[j][None, :]).item()
            )
            assert delta >= threshold, (
                f"pair ({i}, {j}) has delta_e {delta:.2f} < {threshold}"
            )


def test_palette_reports_effective_k():
    img = _near_duplicate_grey_image()
    k = 12
    palette, indices, effective_k = quantize(
        img, k=k, random_state=0, min_delta_e=10.0
    )

    assert effective_k < k
    assert effective_k >= 2
    unique_used = np.unique(indices)
    assert len(unique_used) == effective_k
    assert int(indices.max()) < effective_k


def _four_corners_one_centre_image(size: int = 96) -> np.ndarray:
    """Four equal-area corner colours plus a tiny central "subject" colour.
    With ``k=4`` an unweighted fit fills its budget with the four large
    corner clusters; a centre-weighted fit must sacrifice one corner for
    the central colour.
    """
    img = np.empty((size, size, 3), dtype=np.uint8)
    half = size // 2
    img[:half, :half] = (200, 60, 60)      # top-left   — red
    img[:half, half:] = (60, 200, 60)      # top-right  — green
    img[half:, :half] = (60, 60, 200)      # bottom-left — blue
    img[half:, half:] = (200, 200, 60)     # bottom-right — yellow
    cy = cx = size // 2
    radius = max(2, size // 12)
    img[cy - radius : cy + radius, cx - radius : cx + radius] = (180, 60, 200)  # magenta
    return img


def test_quantize_accepts_sample_weight_and_shifts_centroids():
    img = _four_corners_one_centre_image()
    target = np.array([180, 60, 200], dtype=np.float32)
    k = 4

    pal_unweighted, _, _ = quantize(
        img, k=k, random_state=0, min_delta_e=0.0
    )
    d_unweighted = np.linalg.norm(
        pal_unweighted.astype(np.float32) - target, axis=1
    ).min()

    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    sigma = max(h, w) / 16.0
    weights = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    weights = weights.astype(np.float32)
    pal_weighted, _, _ = quantize(
        img, k=k, random_state=0, min_delta_e=0.0, sample_weight=weights
    )
    d_weighted = np.linalg.norm(
        pal_weighted.astype(np.float32) - target, axis=1
    ).min()

    assert d_unweighted > 60, (
        "fixture sanity: unweighted fit shouldn't reach the magenta subject; "
        f"got distance {d_unweighted:.1f}"
    )
    assert d_weighted < 30, (
        "centroid weighting must pull at least one centroid onto the "
        f"central magenta subject ({target.tolist()}); "
        f"weighted nearest distance={d_weighted:.1f}"
    )


def test_quantize_rejects_wrongly_shaped_sample_weight():
    img = _two_color_image()
    with pytest.raises(ValueError):
        quantize(img, k=2, sample_weight=np.ones(7, dtype=np.float32))
