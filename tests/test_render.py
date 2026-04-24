import numpy as np

from pbn.render import render_palette, render_preview, render_template


def test_render_preview_maps_each_index_to_its_palette_color():
    palette = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        dtype=np.uint8,
    )
    indices = np.array(
        [[0, 1, 2], [2, 0, 1]],
        dtype=np.int32,
    )
    preview = render_preview(palette, indices)

    assert preview.shape == (2, 3, 3)
    assert preview.dtype == np.uint8
    assert tuple(preview[0, 0]) == (255, 0, 0)
    assert tuple(preview[0, 1]) == (0, 255, 0)
    assert tuple(preview[0, 2]) == (0, 0, 255)
    assert tuple(preview[1, 0]) == (0, 0, 255)


def test_render_preview_rejects_out_of_range_index():
    import pytest

    palette = np.zeros((2, 3), dtype=np.uint8)
    indices = np.array([[0, 5]], dtype=np.int32)
    with pytest.raises(IndexError):
        render_preview(palette, indices)


def test_render_template_returns_white_image_with_black_boundaries():
    palette = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    indices = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    template = render_template(indices, palette, scale=4)

    assert template.dtype == np.uint8
    assert template.shape == (3 * 4, 4 * 4, 3)
    # Mostly white canvas.
    white_ratio = (template == 255).all(axis=-1).mean()
    assert white_ratio > 0.5
    # Has some pure-black pixels (the outline between regions).
    assert (template == 0).all(axis=-1).any()


def test_render_template_scale_multiplies_dimensions():
    palette = np.zeros((1, 3), dtype=np.uint8)
    indices = np.zeros((5, 6), dtype=np.int32)
    template = render_template(indices, palette, scale=3)
    assert template.shape == (15, 18, 3)


def test_render_template_draws_digit_in_each_region():
    # Two vertical stripes — we want at least one non-white, non-black pixel
    # introduced by digit rendering somewhere in each region.
    palette = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.uint8)
    indices = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    template = render_template(indices, palette, scale=8)
    # Left half and right half should each contain black pixels (from digits).
    left = template[:, : template.shape[1] // 2]
    right = template[:, template.shape[1] // 2 :]
    assert (left == 0).all(axis=-1).sum() > 0
    assert (right == 0).all(axis=-1).sum() > 0


def test_render_palette_contains_every_swatch_colour():
    palette = np.array(
        [[200, 10, 10], [10, 200, 10], [10, 10, 200]],
        dtype=np.uint8,
    )
    legend = render_palette(palette, swatch_size=20)

    assert legend.dtype == np.uint8
    assert legend.ndim == 3 and legend.shape[2] == 3

    # Every palette colour appears somewhere on the legend.
    pixels = legend.reshape(-1, 3)
    for rgb in palette:
        assert any(np.array_equal(p, rgb) for p in pixels), (
            f"palette colour {tuple(rgb)} not found in legend"
        )


def test_render_palette_height_grows_with_palette_size():
    small = render_palette(np.zeros((2, 3), dtype=np.uint8), swatch_size=16)
    big = render_palette(np.zeros((8, 3), dtype=np.uint8), swatch_size=16)
    assert big.shape[0] > small.shape[0]
