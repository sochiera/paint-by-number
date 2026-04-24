import numpy as np

from pbn.regions import label_regions


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
