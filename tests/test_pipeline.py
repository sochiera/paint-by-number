import numpy as np

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
