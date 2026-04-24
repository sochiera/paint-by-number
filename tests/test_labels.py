import numpy as np

from pbn.labels import label_positions


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
