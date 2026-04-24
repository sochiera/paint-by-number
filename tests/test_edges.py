import numpy as np

from pbn.edges import region_boundaries


def test_solid_image_has_no_boundaries():
    indices = np.zeros((4, 4), dtype=np.int32)
    edges = region_boundaries(indices)
    assert edges.shape == indices.shape
    assert edges.dtype == bool
    assert not edges.any()


def test_vertical_split_marks_both_sides_of_border():
    indices = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.int32,
    )
    edges = region_boundaries(indices)
    # Right edge of left block and left edge of right block are boundaries.
    assert edges[:, 1].all()
    assert edges[:, 2].all()
    # Interior columns that are not next to a colour change are not boundaries.
    assert not edges[:, 0].any()
    assert not edges[:, 3].any()


def test_image_border_is_not_a_boundary():
    indices = np.zeros((3, 3), dtype=np.int32)
    edges = region_boundaries(indices)
    assert not edges.any()


def test_diagonal_neighbours_are_not_adjacent_for_boundary():
    indices = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int32,
    )
    edges = region_boundaries(indices)
    # Every pixel has a 4-neighbour of a different colour → all pixels are
    # boundary pixels.
    assert edges.all()
