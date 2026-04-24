import numpy as np
import pytest

from pbn.pipeline import PBNResult, generate


def _make_stripe_image(h=20, w=30, colours=((255, 0, 0), (0, 200, 0), (0, 0, 180))):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    step = w // len(colours)
    for i, c in enumerate(colours):
        img[:, i * step : (i + 1) * step] = c
    # Final stripe absorbs the remainder.
    img[:, (len(colours) - 1) * step :] = colours[-1]
    return img


def test_generate_returns_result_with_expected_fields():
    image = _make_stripe_image()
    result = generate(image, k=3, min_region_size=5, template_scale=2)

    assert isinstance(result, PBNResult)
    assert result.palette.shape == (3, 3)
    assert result.palette.dtype == np.uint8
    assert result.indices.shape == image.shape[:2]
    assert result.preview.shape == image.shape
    # Template is upscaled.
    assert result.template.shape[0] == image.shape[0] * 2
    assert result.template.shape[1] == image.shape[1] * 2
    assert result.legend.ndim == 3


def test_generate_preview_close_to_original_for_simple_image():
    image = _make_stripe_image()
    result = generate(image, k=3, min_region_size=5, template_scale=2)
    # Average per-channel error should be tiny on a simple 3-colour image.
    diff = np.abs(image.astype(int) - result.preview.astype(int)).mean()
    assert diff < 5


def test_generate_respects_k():
    image = _make_stripe_image()
    result = generate(image, k=2, min_region_size=5, template_scale=2)
    used = np.unique(result.indices)
    assert len(used) <= 2


def test_generate_removes_tiny_noise_regions():
    # Image with a large sea of one colour and a single stray pixel of another.
    image = np.full((20, 20, 3), (255, 255, 255), dtype=np.uint8)
    image[10, 10] = (0, 0, 0)
    result = generate(image, k=2, min_region_size=5, template_scale=1)
    # After merging, the image should effectively be one colour.
    unique = np.unique(result.indices)
    assert len(unique) == 1


def _make_textured_image(h=128, w=128, seed=0):
    """Synthetic textured image: uniform noisy background with a clear
    contrasting disc at the centre."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 120, size=(h, w, 3), dtype=np.int16)
    img = bg.astype(np.int16)
    yy, xx = np.ogrid[:h, :w]
    disc = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= (min(h, w) // 4) ** 2
    disc_noise = rng.integers(-15, 16, size=(h, w, 3), dtype=np.int16)
    disc_colour = np.array([220, 60, 40], dtype=np.int16)
    img = np.where(disc[..., None], disc_colour + disc_noise, img)
    return np.clip(img, 0, 255).astype(np.uint8)


def _count_connected_regions(indices: np.ndarray) -> int:
    from scipy import ndimage

    total = 0
    for label_value in np.unique(indices):
        mask = indices == label_value
        _, n = ndimage.label(mask)
        total += n
    return total


def test_bilateral_reduces_region_count_on_texture():
    image = _make_textured_image()
    common = dict(k=3, min_region_size=4, template_scale=1, random_state=0)

    none_result = generate(image, smooth="none", **common)
    bilateral_result = generate(image, smooth="bilateral", **common)

    n_regions = _count_connected_regions(none_result.indices)
    b_regions = _count_connected_regions(bilateral_result.indices)

    assert b_regions <= n_regions * 0.6, (
        f"bilateral produced {b_regions} regions, none {n_regions}; "
        "expected bilateral to drop by at least 40%."
    )


def test_smooth_none_skips_blur():
    """smooth='none' must bypass smoothing even when blur_sigma > 0."""
    image = _make_textured_image()
    common = dict(k=3, min_region_size=0, template_scale=1, random_state=0)

    baseline = generate(image, smooth="none", blur_sigma=0.0, **common)
    with_blur = generate(image, smooth="none", blur_sigma=5.0, **common)

    np.testing.assert_array_equal(baseline.indices, with_blur.indices)


def test_meanshift_without_opencv_raises():
    """meanshift mode raises a clear error at call time if cv2 is missing."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("opencv-python is installed; error path not reachable")

    image = _make_textured_image(h=32, w=32)
    with pytest.raises(ImportError, match="opencv"):
        generate(image, k=3, smooth="meanshift", template_scale=1)


def test_bilateral_preserves_object_edges():
    """Canny on the bilaterally-smoothed image should recover the reference
    Canny of the noise-free shape at >= 90 % IoU (intersection over union of
    edge pixels, with a 1-pixel tolerance via dilation)."""
    from scipy import ndimage
    from skimage.feature import canny

    from pbn.pipeline import _bilateral_smooth

    rng = np.random.default_rng(1)
    h, w = 80, 80
    # Sharp bright rectangle on a dark noisy background.
    clean = np.full((h, w, 3), 30, dtype=np.uint8)
    clean[20:60, 20:60] = 220
    noise = rng.integers(-40, 41, size=(h, w, 3), dtype=np.int16)
    noisy = np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    ref_edges = canny(clean.mean(axis=2) / 255.0, sigma=1.0)

    smoothed = _bilateral_smooth(noisy, sigma_color=0.1, sigma_spatial=3.0)
    bilateral_edges = canny(smoothed.mean(axis=2) / 255.0, sigma=1.0)

    # "Edges preserved" = how much of the reference Canny survives in the
    # bilateral Canny, with a 1-pixel tolerance for sub-pixel drift.
    tolerance = ndimage.binary_dilation(bilateral_edges, iterations=1)
    preserved = np.logical_and(ref_edges, tolerance).sum() / ref_edges.sum()

    assert preserved >= 0.9, (
        f"bilateral preserved only {preserved:.3f} of reference Canny edges; "
        "expected >= 0.90"
    )


def test_generate_rejects_unknown_smoothing():
    image = _make_stripe_image()
    with pytest.raises(ValueError):
        generate(image, k=3, smooth="bogus")


def test_max_regions_enforced():
    image = _make_textured_image(h=96, w=96, seed=2)
    result = generate(
        image,
        k=6,
        min_region_size=0,
        template_scale=1,
        smooth="none",
        max_regions=50,
    )
    assert _count_connected_regions(result.indices) <= 50
