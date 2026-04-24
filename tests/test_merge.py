import numpy as np

from pbn.regions import label_regions, merge_small_regions


def test_merge_noop_when_all_regions_large_enough():
    indices = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    result = merge_small_regions(indices, min_size=2)
    assert np.array_equal(result, indices)


def test_merge_absorbs_single_stray_pixel_into_surrounding_color():
    # A single "1" pixel surrounded by "0"s should be absorbed.
    indices = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    result = merge_small_regions(indices, min_size=2)
    assert np.all(result == 0)


def test_merge_prefers_neighbour_with_longest_shared_boundary():
    # Small '2' pixel has three 4-neighbours of colour '1' and one of colour
    # '0', so it should be absorbed into '1'.
    indices = np.array(
        [
            [0, 0, 0],
            [1, 2, 1],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    result = merge_small_regions(indices, min_size=2)
    assert result[1, 1] == 1
    assert (result == 2).sum() == 0


def test_merge_leaves_shape_unchanged():
    indices = np.array([[0, 0], [0, 1]], dtype=np.int32)
    result = merge_small_regions(indices, min_size=3)
    assert result.shape == indices.shape


def test_merge_multiple_tiny_regions_all_removed():
    # Two isolated small regions of different colours, inside a large '0' sea.
    indices = np.zeros((7, 7), dtype=np.int32)
    indices[1, 1] = 1
    indices[5, 5] = 2
    result = merge_small_regions(indices, min_size=2)

    # Both stray pixels absorbed.
    assert np.all(result == 0)
    # And the region labelling now sees a single region.
    assert len(np.unique(label_regions(result))) == 1


def test_merge_min_size_one_is_noop():
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 3, (6, 6)).astype(np.int32)
    result = merge_small_regions(indices, min_size=1)
    assert np.array_equal(result, indices)
