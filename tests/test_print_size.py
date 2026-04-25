import pytest

from pbn.print_size import PRINT_SIZES, resolve_print_params


def test_a4_300dpi_landscape_image_picks_landscape_page():
    res = resolve_print_params("A4", dpi=300, image_h=900, image_w=1600)

    # A4 landscape at 300 DPI: ~3508 × 2480 px.
    # 1600 × scale ≤ 3508 and 900 × scale ≤ 2480 ⇒ scale ≤ 2.19, so scale = 2.
    assert res.scale == 2
    # 4 mm² at 300 DPI = 4 * (300/25.4)² ≈ 558 output px.
    # source_px * scale² ≥ 558 ⇒ source_px ≥ 140.
    assert 130 <= res.min_region_size <= 145


def test_a4_300dpi_portrait_image_picks_portrait_page():
    res = resolve_print_params("A4", dpi=300, image_h=1600, image_w=900)
    # Same image, transposed → A4 portrait. Should give the same numbers.
    assert res.scale == 2
    assert 130 <= res.min_region_size <= 145


def test_higher_dpi_raises_min_region_size():
    low = resolve_print_params("A4", dpi=150, image_h=900, image_w=1600)
    high = resolve_print_params("A4", dpi=600, image_h=900, image_w=1600)
    # Higher DPI ⇒ more source-pixels per mm on output ⇒ larger min_region.
    # (At higher DPI the scale also rises, partially compensating, but for
    # the same image the min_region threshold should not shrink.)
    assert high.min_region_size >= low.min_region_size


def test_a3_gives_larger_scale_than_a4():
    a4 = resolve_print_params("A4", dpi=300, image_h=900, image_w=1600)
    a3 = resolve_print_params("A3", dpi=300, image_h=900, image_w=1600)
    assert a3.scale >= a4.scale


def test_unknown_page_raises():
    with pytest.raises(ValueError):
        resolve_print_params("Tabloid", dpi=300, image_h=100, image_w=100)


def test_bad_dpi_raises():
    with pytest.raises(ValueError):
        resolve_print_params("A4", dpi=0, image_h=100, image_w=100)


def test_bad_dimensions_raise():
    with pytest.raises(ValueError):
        resolve_print_params("A4", dpi=300, image_h=0, image_w=100)
    with pytest.raises(ValueError):
        resolve_print_params("A4", dpi=300, image_h=100, image_w=-1)


def test_choices_lists_known_pages():
    assert set(PRINT_SIZES) == {"A4", "A3", "Letter"}


def test_scale_clamped_to_one_for_huge_inputs():
    res = resolve_print_params("A4", dpi=72, image_h=4000, image_w=4000)
    assert res.scale >= 1
