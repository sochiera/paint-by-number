import numpy as np

from pbn.labels import compute_placements, label_positions


def test_single_region_label_sits_away_from_image_border():
    labels = np.zeros((5, 7), dtype=np.int32)
    positions = label_positions(labels)
    assert len(positions) == 1
    x, y, lbl = positions[0]
    assert lbl == 0
    # Must be at least two pixels from every edge (the true centre of a 5x7).
    H, W = labels.shape
    assert 2 <= y <= H - 3
    assert 2 <= x <= W - 3


def test_two_regions_produce_two_positions():
    labels = np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    positions = label_positions(labels)
    assert len(positions) == 2
    by_label = {p[2]: (p[0], p[1]) for p in positions}
    assert set(by_label) == {0, 1}
    # Label 0 position must fall inside the 0 region.
    x, y = by_label[0]
    assert labels[y, x] == 0
    x, y = by_label[1]
    assert labels[y, x] == 1


def test_label_positions_fall_inside_own_region():
    rng = np.random.default_rng(7)
    from pbn.regions import label_regions

    indices = rng.integers(0, 4, (12, 12)).astype(np.int32)
    labels = label_regions(indices)
    for x, y, lbl in label_positions(labels):
        assert labels[y, x] == lbl


def test_thin_region_still_gets_one_position():
    # A long 1-pixel-wide column stuck inside a sea of 0s.
    labels = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.int32,
    )
    positions = label_positions(labels)
    by_label = {p[2]: (p[0], p[1]) for p in positions}
    assert 1 in by_label
    x, y = by_label[1]
    assert labels[y, x] == 1


def _make_indices_with_tiny_and_big(shape=(40, 40)):
    """Return an indices map with a large surrounding region (index 0) and a
    tiny 2x2 region (index 1) at a known location."""
    indices = np.zeros(shape, dtype=np.int32)
    # Tiny 2x2 block of label 1 near the centre-ish, surrounded by label 0.
    indices[18:20, 18:20] = 1
    return indices


def test_small_region_uses_leadline():
    indices = _make_indices_with_tiny_and_big()
    placements = compute_placements(indices, scale=1)

    # Find the placement for the tiny region (palette index 1).
    tiny = [p for p in placements if p["palette_index"] == 1]
    assert len(tiny) == 1
    p = tiny[0]

    # Tiny region must be flagged for a lead line.
    assert p["leadline_from"] is not None
    assert p["leadline_to"] is not None

    # And the digit itself must be placed outside the tiny region's pixels.
    y, x = p["digit_pos"]
    assert indices[y, x] != 1


def test_digit_size_scales_with_inscribed_radius():
    # Big round-ish region (label 0) next to a medium square (label 1).
    indices = np.zeros((60, 120), dtype=np.int32)
    # Large region: a big rectangle on the left spanning almost the whole
    # height, so the inscribed radius is large.
    indices[5:55, 5:55] = 0  # already 0 — kept for clarity
    # Medium region: a small square on the right.
    indices[25:33, 80:88] = 1

    placements = compute_placements(indices, scale=1)
    by_idx = {p["palette_index"]: p for p in placements}
    assert 0 in by_idx and 1 in by_idx

    # The large region's digit size must strictly exceed the small region's.
    assert by_idx[0]["digit_size"] > by_idx[1]["digit_size"]


def test_digit_anchor_inside_region_when_fits():
    # A single 30x30 region filling the whole image; digit must go inside.
    indices = np.zeros((30, 30), dtype=np.int32)
    placements = compute_placements(indices, scale=2)
    assert len(placements) == 1
    p = placements[0]
    y, x = p["digit_pos"]
    assert indices[y, x] == 0
    assert p["leadline_from"] is None
    assert p["leadline_to"] is None
