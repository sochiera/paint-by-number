from __future__ import annotations

import heapq
from collections import Counter

import numpy as np
from scipy import ndimage


# 4-connectivity cross
_STRUCTURE_4 = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=bool,
)

_NEIGHBOUR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def label_regions(indices: np.ndarray) -> np.ndarray:
    """Label 4-connected regions of equal palette index.

    Returns an ``(H, W) int32`` array with labels numbered contiguously
    from 0 to ``N - 1``.
    """
    if indices.ndim != 2:
        raise ValueError(f"expected 2D indices, got shape {indices.shape}")

    out = np.zeros_like(indices, dtype=np.int32)
    next_label = 0
    for value in np.unique(indices):
        mask = indices == value
        labelled, count = ndimage.label(mask, structure=_STRUCTURE_4)
        # Shift labels so they do not collide with previously assigned ones.
        shifted = np.where(labelled > 0, labelled + next_label - 1, 0)
        out[mask] = shifted[mask]
        next_label += count
    return out


def merge_small_regions(indices: np.ndarray, min_size: int) -> np.ndarray:
    """Absorb each 4-connected region smaller than ``min_size`` into the
    neighbouring region with the longest shared boundary.

    Iterates until every surviving region has at least ``min_size`` pixels,
    or no more progress can be made (e.g. an image made entirely of a single
    colour that is itself below the threshold).
    """
    if min_size <= 1:
        return indices.copy()

    result = indices.copy()
    h, w = result.shape
    while True:
        labels = label_regions(result)
        sizes = np.bincount(labels.ravel())
        small_labels = np.where(sizes < min_size)[0]
        if len(small_labels) == 0:
            break

        progress = False
        for lbl in small_labels:
            # The label map may be stale after an earlier merge in this pass;
            # re-check size each time.
            mask = labels == lbl
            if not mask.any():
                continue

            neighbour_counts: Counter[int] = Counter()
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                for dy, dx in _NEIGHBOUR_OFFSETS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] != lbl:
                        neighbour_counts[int(result[ny, nx])] += 1

            if not neighbour_counts:
                # Fully enclosed by the image border and contains the whole
                # image — nothing to merge into.
                continue

            best_colour = max(
                neighbour_counts,
                key=lambda c: (neighbour_counts[c], -c),
            )
            result[mask] = best_colour
            progress = True

        if not progress:
            break
    return result


def _component_adjacency(
    labels: np.ndarray,
) -> dict[int, Counter[int]]:
    """Return ``{comp_id: Counter(neighbour_comp_id -> shared_boundary)}``.

    Boundary is counted as the number of 4-connected pixel pairs whose labels
    differ; each such pair contributes 1 to both endpoints' counters.
    """
    adjacency: dict[int, Counter[int]] = {}
    # Vertical neighbours.
    top = labels[:-1, :]
    bottom = labels[1:, :]
    diff = top != bottom
    if diff.any():
        a = top[diff]
        b = bottom[diff]
        for u, v in zip(a.tolist(), b.tolist()):
            adjacency.setdefault(u, Counter())[v] += 1
            adjacency.setdefault(v, Counter())[u] += 1
    # Horizontal neighbours.
    left = labels[:, :-1]
    right = labels[:, 1:]
    diff = left != right
    if diff.any():
        a = left[diff]
        b = right[diff]
        for u, v in zip(a.tolist(), b.tolist()):
            adjacency.setdefault(u, Counter())[v] += 1
            adjacency.setdefault(v, Counter())[u] += 1
    return adjacency


def merge_to_target_count(indices: np.ndarray, max_regions: int) -> np.ndarray:
    """Iteratively merge the smallest 4-connected region into its neighbour
    with the longest shared border until the total number of components
    is ``<= max_regions``.

    Tiebreaks (deterministic): longest shared border, then largest neighbour
    size, then lowest palette index.

    The palette-index dtype and shape are preserved; no new palette entries
    are created (a merged component simply inherits the winning neighbour's
    palette index).
    """
    if indices.ndim != 2:
        raise ValueError(f"expected 2D indices, got shape {indices.shape}")
    if max_regions < 1:
        raise ValueError(f"max_regions must be >= 1, got {max_regions}")

    result = indices.copy()

    # Recompute component labels each outer iteration; a single iteration
    # processes many merges by keeping a heap of current sizes. Stale heap
    # entries are discarded lazily via the component-size snapshot.
    while True:
        labels = label_regions(result)
        n_components = int(labels.max()) + 1
        if n_components <= max_regions:
            break

        sizes = np.bincount(labels.ravel())
        # Track each component's representative pixel so we can read its
        # palette index quickly even after mutations.
        flat = labels.ravel()
        first_index = np.full(n_components, -1, dtype=np.int64)
        # np.unique with return_index gives the first occurrence of each label
        # in a single C-level pass.
        uniq, first = np.unique(flat, return_index=True)
        first_index[uniq] = first
        palette_of = result.ravel()[first_index]

        adjacency = _component_adjacency(labels)

        # Build initial heap keyed by (size, palette_index, comp_id).
        heap: list[tuple[int, int, int]] = [
            (int(sizes[c]), int(palette_of[c]), int(c))
            for c in range(n_components)
        ]
        heapq.heapify(heap)

        # Union-find: each component can be absorbed into another within this
        # pass, avoiding O(H*W) relabelling per merge.
        parent = list(range(n_components))
        current_size = sizes.astype(np.int64).copy()
        current_palette = palette_of.astype(np.int64).copy()
        alive = np.ones(n_components, dtype=bool)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merges_done = 0
        target_merges = n_components - max_regions

        while merges_done < target_merges and heap:
            size, pal, comp = heapq.heappop(heap)
            root = find(comp)
            if not alive[root]:
                continue
            # Skip stale entries whose size or palette index has changed.
            if size != current_size[root] or pal != current_palette[root]:
                continue

            # Find the best neighbour among the currently alive roots.
            neighbour_borders: Counter[int] = Counter()
            for nbr, border in adjacency.get(comp, {}).items():
                nbr_root = find(nbr)
                if nbr_root == root or not alive[nbr_root]:
                    continue
                neighbour_borders[nbr_root] += border

            if not neighbour_borders:
                # Isolated (shouldn't happen unless single component). Give up
                # on this one and continue — the outer while will re-label.
                alive[root] = False
                continue

            best = max(
                neighbour_borders.items(),
                key=lambda item: (
                    item[1],
                    current_size[item[0]],
                    -int(current_palette[item[0]]),
                ),
            )
            target_root = best[0]

            # Merge `root` into `target_root`.
            parent[root] = target_root
            alive[root] = False
            current_size[target_root] += current_size[root]
            # Fold root's adjacency into target_root.
            merged_adj = adjacency.setdefault(target_root, Counter())
            for nbr, border in adjacency.get(comp, {}).items():
                nbr_root = find(nbr)
                if nbr_root == target_root:
                    continue
                merged_adj[nbr_root] += border
                # Update the neighbour's view too so later iterations see
                # the combined border towards the merged component.
                nbr_entry = adjacency.setdefault(nbr_root, Counter())
                nbr_entry[target_root] = nbr_entry.get(target_root, 0) + border
                nbr_entry.pop(comp, None)
                nbr_entry.pop(root, None)
            merged_adj.pop(target_root, None)
            merged_adj.pop(comp, None)
            merged_adj.pop(root, None)

            # Re-push the target with its new size so the heap can see it
            # if it becomes the smallest at some point.
            heapq.heappush(
                heap,
                (
                    int(current_size[target_root]),
                    int(current_palette[target_root]),
                    int(target_root),
                ),
            )
            merges_done += 1

        # Apply the merges to `result` by remapping palette indices.
        # Each original component maps to its root's palette index.
        comp_to_pal = np.empty(n_components, dtype=result.dtype)
        for c in range(n_components):
            comp_to_pal[c] = current_palette[find(c)]
        result = comp_to_pal[labels]

        if merges_done == 0:
            # No progress — avoid an infinite loop.
            break

    return result
