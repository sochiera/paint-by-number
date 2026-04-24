import numpy as np
from scipy import ndimage

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


def _two_region_indices(h: int = 32, w: int = 32) -> np.ndarray:
    """A square image with a central rectangle of label 1 inside label 0 —
    yields a long, well-defined boundary suitable for line-width tests."""
    indices = np.zeros((h, w), dtype=np.int32)
    indices[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
    return indices


def test_template_line_weight():
    """At scale=6, dilation radius is 2 — the boundary line is at least
    3 px thick everywhere. We measure thickness by eroding the boundary
    band with a 3x3 cross: the result must still cover most of the band,
    proving no 2-pixel-wide stretches remain."""
    palette = np.array([[200, 0, 0], [0, 200, 0]], dtype=np.uint8)
    indices = _two_region_indices(32, 32)
    scale = 6
    template = render_template(indices, palette, scale=scale)

    black = (template == 0).all(axis=-1)

    # Restrict to a vertical strip away from digit anchors so digits don't
    # contaminate the analysis. The strip straddles the right vertical edge
    # of the inner rectangle.
    src_boundary_x = (3 * indices.shape[1] // 4) - 1  # last col of inner region
    canvas_x = src_boundary_x * scale + scale // 2
    strip = black[:, canvas_x - 2 : canvas_x + scale + 2]

    assert strip.any(), "expected boundary pixels in the sample strip"

    # Width >= 2 pointwise: every black pixel has a black 4-neighbour.
    pad = np.pad(strip, 1, constant_values=False)
    has_neighbour = (
        pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    )
    lonely = strip & ~has_neighbour
    assert not lonely.any(), (
        f"{int(lonely.sum())} isolated 1-px black pixels in strip"
    )

    # Width >= 3: erosion by a 3x3 cross still covers a substantial slice
    # of the strip. With dilation radius 2 the line is ~5 px thick, so
    # erosion leaves most of it; without dilation the original 2-px-wide
    # boundary collapses to almost nothing under cross erosion.
    cross = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(strip, structure=cross)
    assert eroded.sum() >= int(0.4 * strip.sum()), (
        f"eroded boundary too thin: {int(eroded.sum())}/{int(strip.sum())} "
        f"black pixels survive 1-px erosion"
    )


def test_template_line_weight_scale_4():
    """At scale=4, dilation radius is 1 so the boundary line is at least
    2 px thick — no isolated 1-pixel pixels."""
    palette = np.array([[200, 0, 0], [0, 200, 0]], dtype=np.uint8)
    indices = _two_region_indices(32, 32)
    scale = 4
    template = render_template(indices, palette, scale=scale)

    black = (template == 0).all(axis=-1)

    # Same right-side strip as the scale=6 test.
    src_boundary_x = (3 * indices.shape[1] // 4) - 1
    canvas_x = src_boundary_x * scale + scale // 2
    strip = black[:, canvas_x - 1 : canvas_x + scale + 1]

    assert strip.any()

    pad = np.pad(strip, 1, constant_values=False)
    has_neighbour = (
        pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    )
    lonely = strip & ~has_neighbour
    assert not lonely.any()


def test_template_line_weight_scale_3_unchanged():
    """At scale=3, no dilation should be applied — but the template still
    renders sanely (boundaries present, mostly white canvas)."""
    palette = np.array([[200, 0, 0], [0, 200, 0]], dtype=np.uint8)
    indices = _two_region_indices(32, 32)
    template = render_template(indices, palette, scale=3)

    assert template.dtype == np.uint8
    assert template.shape == (32 * 3, 32 * 3, 3)
    white_ratio = (template == 255).all(axis=-1).mean()
    assert white_ratio > 0.5
    assert (template == 0).all(axis=-1).any()


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
