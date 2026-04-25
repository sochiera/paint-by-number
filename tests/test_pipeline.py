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


def _make_noisy_with_object(h=128, w=128, seed=3):
    """Noisy textured dark background with a large bright central object."""
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 80, size=(h, w, 3), dtype=np.int16)
    yy, xx = np.ogrid[:h, :w]
    # Large central disc — size big enough that cleanup preserves it.
    disc = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= (min(h, w) // 3) ** 2
    disc_noise = rng.integers(-10, 11, size=(h, w, 3), dtype=np.int16)
    disc_colour = np.array([240, 240, 240], dtype=np.int16)
    img = np.where(disc[..., None], disc_colour + disc_noise, img)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_majority_cleanup_reduces_components():
    image = _make_noisy_with_object()
    common = dict(
        k=5,
        min_region_size=1,
        template_scale=1,
        smooth="none",
        random_state=0,
    )
    none_result = generate(image, cleanup="none", **common)
    maj_result = generate(image, cleanup="majority", **common)

    n_none = _count_connected_regions(none_result.indices)
    n_maj = _count_connected_regions(maj_result.indices)

    assert n_maj <= n_none * 0.7, (
        f"majority cleanup produced {n_maj} regions, none {n_none}; "
        "expected majority to drop by at least 30%."
    )


def test_majority_cleanup_preserves_main_object():
    from scipy import ndimage
    from skimage.feature import canny

    image = _make_noisy_with_object()
    common = dict(
        k=5,
        min_region_size=1,
        template_scale=1,
        smooth="none",
        random_state=0,
    )
    none_result = generate(image, cleanup="none", **common)
    maj_result = generate(image, cleanup="majority", **common)

    # Compare Canny edges of the quantised preview; the object's contour
    # should remain essentially the same. Restrict to a band around the
    # central disc so we measure the object's contour, not background
    # speckle that cleanup is expected to dissolve.
    none_edges = canny(none_result.preview.mean(axis=2) / 255.0, sigma=1.0)
    maj_edges = canny(maj_result.preview.mean(axis=2) / 255.0, sigma=1.0)

    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    r_outer = min(h, w) // 3 + 4
    r_inner = min(h, w) // 3 - 4
    dist2 = (yy - h // 2) ** 2 + (xx - w // 2) ** 2
    band = (dist2 <= r_outer**2) & (dist2 >= r_inner**2)

    # Preservation: how much of the original (cleanup-off) contour
    # survives in the cleanup-on contour with a 1-pixel tolerance for
    # sub-pixel drift. Same style as test_bilateral_preserves_object_edges.
    ref_band = none_edges & band
    maj_tolerance = ndimage.binary_dilation(maj_edges & band, iterations=1)
    preserved = np.logical_and(ref_band, maj_tolerance).sum() / max(
        ref_band.sum(), 1
    )

    assert preserved >= 0.9, (
        f"main-object contour preservation was {preserved:.3f}; expected >= 0.90"
    )


def test_cleanup_none_is_noop():
    """cleanup='none' must produce identical indices to prior behaviour
    (no majority filter applied between quantise and merge)."""
    from pbn.quantize import quantize

    image = _make_noisy_with_object()
    # Replicate what generate() does internally with cleanup='none' and
    # no merging: a bare quantisation output.
    _, expected_indices, _ = quantize(image, k=5, random_state=0)

    result = generate(
        image,
        k=5,
        min_region_size=0,
        template_scale=1,
        smooth="none",
        cleanup="none",
        random_state=0,
    )
    np.testing.assert_array_equal(result.indices, expected_indices)


def test_generate_rejects_unknown_cleanup():
    image = _make_stripe_image()
    with pytest.raises(ValueError):
        generate(image, k=3, cleanup="bogus")


def _dark_dominant_with_central_subject(size: int = 96) -> np.ndarray:
    """Background of varied dark tones (~96 % of pixels) plus a small
    central magenta subject. Without saliency the K-means budget is spent
    on dark tones and the subject blends; with ``saliency='center'`` at
    least one centroid lands on it.
    """
    img = np.empty((size, size, 3), dtype=np.uint8)
    half = size // 2
    img[:half, :half] = (200, 60, 60)
    img[:half, half:] = (60, 200, 60)
    img[half:, :half] = (60, 60, 200)
    img[half:, half:] = (200, 200, 60)
    cy = cx = size // 2
    radius = max(2, size // 12)
    img[cy - radius : cy + radius, cx - radius : cx + radius] = (180, 60, 200)
    return img


def test_saliency_center_shifts_palette_towards_subject():
    image = _dark_dominant_with_central_subject()
    target = np.array([180, 60, 200], dtype=np.float32)

    result_none = generate(
        image, k=4, min_region_size=0, smooth="none", cleanup="none",
        min_delta_e=0.0, template_scale=1, saliency="none",
    )
    result_center = generate(
        image, k=4, min_region_size=0, smooth="none", cleanup="none",
        min_delta_e=0.0, template_scale=1, saliency="center",
    )

    def _closest_distance(palette: np.ndarray, k: int) -> float:
        return float(
            np.linalg.norm(palette[:k].astype(np.float32) - target, axis=1).min()
        )

    d_none = _closest_distance(result_none.palette, result_none.effective_k)
    d_center = _closest_distance(
        result_center.palette, result_center.effective_k
    )
    assert d_none > 60
    assert d_center < d_none - 15, (
        f"saliency='center' should pull a centroid materially closer to "
        f"{target.tolist()}; closest was {d_center:.1f} vs none={d_none:.1f}"
    )


def test_saliency_none_matches_baseline():
    image = _make_stripe_image()
    base = generate(image, k=3, min_region_size=5, template_scale=1)
    explicit = generate(
        image, k=3, min_region_size=5, template_scale=1, saliency="none"
    )
    assert np.array_equal(base.palette, explicit.palette)
    assert np.array_equal(base.indices, explicit.indices)


def test_generate_rejects_unknown_saliency():
    image = _make_stripe_image()
    with pytest.raises(ValueError):
        generate(image, k=3, saliency="bogus")


def _speckled_image(size: int = 80, seed: int = 0) -> np.ndarray:
    """Solid-blue background with many isolated red specks. After quantize
    + cleanup the red specks survive as many small components of one
    colour, which is exactly what max_per_color is meant to suppress."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (40, 60, 200), dtype=np.uint8)
    placed = 0
    while placed < 60:
        y = int(rng.integers(2, size - 2))
        x = int(rng.integers(2, size - 2))
        img[y, x] = (220, 30, 30)
        placed += 1
    return img


def _component_count_per_color(indices: np.ndarray) -> dict[int, int]:
    from pbn.regions import label_regions  # noqa: PLC0415

    labels = label_regions(indices)
    flat = labels.ravel()
    palette_flat = indices.ravel()
    first = np.unique(flat, return_index=True)[1]
    out: dict[int, int] = {}
    for pal in palette_flat[first].tolist():
        out[int(pal)] = out.get(int(pal), 0) + 1
    return out


def test_max_per_color_caps_fragments_in_pipeline():
    image = _speckled_image()
    base = generate(
        image, k=3, min_region_size=0, smooth="none", cleanup="none",
        min_delta_e=0.0, template_scale=1, max_regions=None,
    )
    capped = generate(
        image, k=3, min_region_size=0, smooth="none", cleanup="none",
        min_delta_e=0.0, template_scale=1, max_regions=None,
        max_per_color=5,
    )
    base_counts = _component_count_per_color(base.indices)
    capped_counts = _component_count_per_color(capped.indices)

    assert max(base_counts.values()) > 10, (
        f"fixture sanity: expected over-fragmented colour; got {base_counts}"
    )
    assert all(n <= 5 for n in capped_counts.values()), (
        f"max_per_color=5 violated: {capped_counts}"
    )


def test_max_per_color_none_is_noop():
    image = _make_stripe_image()
    base = generate(image, k=3, min_region_size=5, template_scale=1)
    explicit = generate(
        image, k=3, min_region_size=5, template_scale=1, max_per_color=None
    )
    np.testing.assert_array_equal(base.indices, explicit.indices)
