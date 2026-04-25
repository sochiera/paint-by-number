import numpy as np
import pytest

from pbn.regions import (
    cap_fragments_per_color,
    label_regions,
    merge_to_target_count,
)


def _count_components(indices: np.ndarray) -> int:
    return int(label_regions(indices).max()) + 1


def test_label_regions_single_color():
    indices = np.zeros((3, 3), dtype=np.int32)
    labels = label_regions(indices)
    assert labels.shape == (3, 3)
    assert np.unique(labels).tolist() == [0]


def test_label_regions_split_into_two_blocks():
    indices = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    labels = label_regions(indices)
    assert set(np.unique(labels).tolist()) == {0, 1}
    # each block is one region
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 2] == labels[1, 3]
    assert labels[0, 0] != labels[0, 2]


def test_label_regions_same_color_but_disconnected_get_different_labels():
    indices = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )
    labels = label_regions(indices)
    # four corners each have colour 0 but are disconnected => 4 regions,
    # plus a single cross-shaped 1-region => 5 total.
    assert len(np.unique(labels)) == 5


def test_label_regions_uses_4_connectivity_not_8():
    # Two color-0 cells touching only diagonally should be 2 regions, not 1.
    indices = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int32,
    )
    labels = label_regions(indices)
    assert labels[0, 0] != labels[1, 1]
    # plus the two 1-cells are also diagonal, also separate
    assert len(np.unique(labels)) == 4


def test_label_regions_labels_are_contiguous_from_zero():
    indices = np.array(
        [
            [2, 2, 5],
            [2, 5, 5],
        ],
        dtype=np.int32,
    )
    labels = label_regions(indices)
    unique = sorted(np.unique(labels).tolist())
    assert unique == list(range(len(unique)))


def _make_scattered_indices(
    h: int = 40, w: int = 40, k: int = 4, seed: int = 0
) -> np.ndarray:
    """Background of 0 with ~30 tiny isolated patches of colours 1..k-1."""
    rng = np.random.default_rng(seed)
    indices = np.zeros((h, w), dtype=np.int32)
    placed = 0
    attempts = 0
    positions: list[tuple[int, int]] = []
    while placed < 30 and attempts < 1000:
        y = int(rng.integers(1, h - 2))
        x = int(rng.integers(1, w - 2))
        attempts += 1
        # Keep patches apart so each stays its own component.
        if any(abs(y - py) <= 2 and abs(x - px) <= 2 for py, px in positions):
            continue
        colour = int(rng.integers(1, k))
        indices[y, x] = colour
        positions.append((y, x))
        placed += 1
    return indices


def test_merge_to_target_count():
    indices = _make_scattered_indices()
    k = int(indices.max()) + 1
    before = _count_components(indices)
    assert before > 5, "synthetic fixture should expose many components"

    merged = merge_to_target_count(indices, max_regions=5)

    assert merged.shape == indices.shape
    assert merged.dtype == indices.dtype
    after = _count_components(merged)
    assert after <= 5, f"expected <= 5 components, got {after}"
    assert merged.min() >= 0
    assert merged.max() < k


def test_merge_to_target_count_noop_when_under_target():
    indices = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    merged = merge_to_target_count(indices, max_regions=10)
    assert merged.shape == indices.shape
    assert merged.dtype == indices.dtype
    np.testing.assert_array_equal(merged, indices)


def test_merge_to_target_count_prefers_longest_boundary():
    # Layout: a tiny '2' patch in the middle touches '0' on one pixel
    # (top) and '1' on three pixels (left, right, bottom). The smallest
    # component (the '2') must be absorbed into '1' (longest boundary).
    indices = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    # Components: one big '0' block (top), '1' block (rest), one stray '2'.
    merged = merge_to_target_count(indices, max_regions=2)
    assert _count_components(merged) <= 2
    # The '2' pixel should have been rewritten to '1' (3 shared border pixels
    # against '1' vs 1 against '0').
    assert merged[2, 2] == 1


def test_merge_to_target_count_deterministic():
    indices = _make_scattered_indices(seed=7)
    a = merge_to_target_count(indices, max_regions=4)
    b = merge_to_target_count(indices, max_regions=4)
    np.testing.assert_array_equal(a, b)


def _components_per_color(indices: np.ndarray) -> dict[int, int]:
    """Map ``palette_index -> number of 4-connected components of that index``."""
    labels = label_regions(indices)
    flat_labels = labels.ravel()
    flat_idx = indices.ravel()
    # For each component label, the palette index is the same everywhere.
    first_pos = np.unique(flat_labels, return_index=True)[1]
    palette_of_label = flat_idx[first_pos]
    counts: dict[int, int] = {}
    for pal in palette_of_label.tolist():
        counts[int(pal)] = counts.get(int(pal), 0) + 1
    return counts


def _make_overfragmented_color_indices() -> np.ndarray:
    """A 14×14 grid: a single big colour-0 background, a couple of colour-1
    shapes (large enough to keep) and many isolated colour-2 specks scattered
    across the background. Colour 2 should have many components; capping it
    must repaint the smallest specks to colour 0 (the only adjacent colour
    for a single isolated speck)."""
    indices = np.zeros((14, 14), dtype=np.int32)
    # Two healthy colour-1 blocks.
    indices[1:5, 1:5] = 1
    indices[9:13, 9:13] = 1
    # Many isolated colour-2 specks.
    speck_positions = [
        (1, 7), (1, 9), (1, 11),
        (3, 7), (3, 11), (3, 13),
        (5, 7), (5, 9),
        (7, 1), (7, 5), (7, 7), (7, 9), (7, 11),
        (9, 1), (9, 3), (9, 5), (9, 7),
        (11, 1), (11, 5), (11, 7),
        (13, 1), (13, 3), (13, 5), (13, 7), (13, 9), (13, 11),
    ]
    for y, x in speck_positions:
        indices[y, x] = 2
    return indices


def test_cap_fragments_per_color_caps_count():
    indices = _make_overfragmented_color_indices()
    before = _components_per_color(indices)
    assert before.get(2, 0) > 5, (
        f"fixture sanity: colour 2 should be heavily fragmented, got {before}"
    )

    capped = cap_fragments_per_color(indices, max_per_color=5)

    after = _components_per_color(capped)
    assert after.get(2, 0) <= 5, (
        f"colour 2 still over cap: {after.get(2, 0)} components"
    )


def test_cap_fragments_per_color_preserves_large_regions():
    indices = _make_overfragmented_color_indices()
    capped = cap_fragments_per_color(indices, max_per_color=5)

    # The two big colour-1 blocks must survive (they're already under cap).
    after = _components_per_color(capped)
    assert after.get(1, 0) == 2

    # Original colour-1 pixels should not have flipped colour.
    assert capped[2, 2] == 1
    assert capped[10, 10] == 1


def test_cap_fragments_per_color_noop_when_under_cap():
    indices = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    capped = cap_fragments_per_color(indices, max_per_color=10)
    np.testing.assert_array_equal(capped, indices)


def test_cap_fragments_per_color_rejects_bad_arg():
    indices = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(ValueError):
        cap_fragments_per_color(indices, max_per_color=0)
    with pytest.raises(ValueError):
        cap_fragments_per_color(indices, max_per_color=-3)


def test_cap_fragments_per_color_deterministic():
    indices = _make_overfragmented_color_indices()
    a = cap_fragments_per_color(indices, max_per_color=4)
    b = cap_fragments_per_color(indices, max_per_color=4)
    np.testing.assert_array_equal(a, b)
