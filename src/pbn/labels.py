from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy import ndimage

from pbn.regions import label_regions


# Heuristic constants for adaptive digit sizing.
_DIGIT_SIZE_COEFF = 1.2
_DIGIT_SIZE_MIN = 6
_DIGIT_SIZE_MAX = 48
# Below this effective (post-scale) inscribed radius the digit won't fit —
# place it outside the region with a lead-line.
_LEADLINE_RADIUS_THRESHOLD_PX = 8


class Placement(TypedDict):
    region_id: int
    palette_index: int
    digit_pos: tuple[int, int]  # (y, x) in source-resolution coordinates
    digit_size: int  # pixel height of the digit at render resolution
    leadline_from: tuple[int, int] | None  # (y, x) at source resolution
    leadline_to: tuple[int, int] | None  # (y, x) at source resolution


def label_positions(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """For every connected region in ``labels`` return a ``(x, y, label)``
    tuple where ``(x, y)`` is the pixel inside that region furthest from
    any region boundary — i.e. the safest spot to print the number.
    """
    if labels.ndim != 2:
        raise ValueError(f"expected 2D labels, got shape {labels.shape}")

    results: list[tuple[int, int, int]] = []
    for lbl in np.unique(labels):
        mask = labels == lbl
        padded = np.pad(mask, 1, constant_values=False)
        dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
        flat_idx = int(np.argmax(dist))
        y, x = np.unravel_index(flat_idx, dist.shape)
        results.append((int(x), int(y), int(lbl)))
    return results


def _inscribed_circle(mask: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Return ``(radius, (y, x))`` of the maximum inscribed circle in a binary
    mask. The pixel coordinate is the argmax of the distance transform over
    the padded mask (so pixels adjacent to the image border are correctly
    penalised)."""
    padded = np.pad(mask, 1, constant_values=False)
    dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    flat_idx = int(np.argmax(dist))
    y, x = np.unravel_index(flat_idx, dist.shape)
    return float(dist[y, x]), (int(y), int(x))


def _pick_leadline_anchor(
    region_mask: np.ndarray,
    indices: np.ndarray,
    region_centre: tuple[int, int],
    required_radius: float,
) -> tuple[int, int] | None:
    """Pick a pixel outside ``region_mask`` that has at least
    ``required_radius`` clearance within its own region and is reasonably
    close to ``region_centre``.

    Returns ``(y, x)`` or ``None`` if no suitable anchor can be found.
    Ties on the chosen score are broken by lowest ``(y, x)``.
    """
    h, w = indices.shape
    # For every pixel outside the target region, compute its clearance to the
    # nearest non-same-label pixel in its own region — this is the maximum
    # digit half-size we could place there.
    clearance = np.zeros_like(indices, dtype=np.float32)
    for lbl in np.unique(indices[~region_mask]):
        other = (indices == lbl) & (~region_mask)
        if not other.any():
            continue
        padded = np.pad(other, 1, constant_values=False)
        dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
        clearance = np.where(other, dist, clearance)

    roomy = clearance >= required_radius
    if not roomy.any():
        # Relax: accept any outside pixel with at least 1 px clearance.
        roomy = clearance >= 1.0
        if not roomy.any():
            return None

    ys, xs = np.where(roomy)
    cy, cx = region_centre
    # Score by distance to region centre (prefer close), then negative
    # clearance (prefer roomier), then (y, x) for deterministic tie-break.
    dists = (ys - cy).astype(np.float64) ** 2 + (xs - cx).astype(np.float64) ** 2
    clr = clearance[ys, xs]
    # Round distances to avoid floating-point jitter in tie-breaks; clearance
    # is also rounded to two decimals.
    order = np.lexsort(
        (
            xs,
            ys,
            -np.round(clr, 2),
            np.round(dists, 3),
        )
    )
    best = int(order[0])
    return int(ys[best]), int(xs[best])


def compute_placements(
    indices: np.ndarray,
    scale: int = 1,
) -> list[Placement]:
    """Compute a digit-placement plan for every 4-connected region in
    ``indices``.

    Each placement entry carries the pixel anchor for the digit, the digit
    pixel height (adaptive to the region's inscribed circle), and, for tiny
    regions that cannot fit their digit, an optional lead-line anchor pair.

    Coordinates are at the source (``indices``) resolution. The caller is
    expected to multiply by ``scale`` when rendering onto the upscaled canvas.
    """
    if indices.ndim != 2:
        raise ValueError(f"expected 2D indices, got shape {indices.shape}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    labels = label_regions(indices)
    placements: list[Placement] = []

    for region_id in np.unique(labels):
        mask = labels == region_id
        if not mask.any():
            continue

        radius, (cy, cx) = _inscribed_circle(mask)
        palette_index = int(indices[cy, cx])

        # Desired digit height at render resolution.
        digit_height = int(
            round(_DIGIT_SIZE_COEFF * radius * scale)
        )
        digit_height = max(_DIGIT_SIZE_MIN, min(_DIGIT_SIZE_MAX, digit_height))

        effective_radius = radius * scale
        needs_leadline = effective_radius < _LEADLINE_RADIUS_THRESHOLD_PX

        if needs_leadline:
            # Required clearance in source-pixel units for a digit of
            # ``_DIGIT_SIZE_MIN`` at the given ``scale``. Half the glyph
            # height (in source pixels) is what we really need.
            required_radius_src = (_DIGIT_SIZE_MIN / 2.0) / scale
            anchor = _pick_leadline_anchor(
                mask, indices, (cy, cx), required_radius_src
            )
            if anchor is not None:
                ay, ax = anchor
                placements.append(
                    Placement(
                        region_id=int(region_id),
                        palette_index=palette_index,
                        digit_pos=(ay, ax),
                        digit_size=digit_height,
                        leadline_from=(ay, ax),
                        leadline_to=(cy, cx),
                    )
                )
                continue
            # Fall through: no viable outside anchor; keep digit inside.

        placements.append(
            Placement(
                region_id=int(region_id),
                palette_index=palette_index,
                digit_pos=(cy, cx),
                digit_size=digit_height,
                leadline_from=None,
                leadline_to=None,
            )
        )

    return placements
