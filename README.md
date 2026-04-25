# paint-by-number

Turn a photograph into a paint-by-number kit: a colour preview, a printable
template with numbered regions, and a palette legend.

## Pipeline

1. Load the input image (any format Pillow can read — PNG, JPEG, …).
2. Edge-preserving smoothing: bilateral, mean-shift, Gaussian, or none.
3. K-means colour quantisation in CIELab so the palette is perceptually
   diverse. Centroid pairs closer than `--min-delta-e` are collapsed and
   K-means is re-run with `k-1` until all remaining pairs clear the
   threshold (or only two colours remain).
4. Majority filter (3×3) over the label map to dissolve isolated speckles
   without moving strong contours.
5. Connected-component labelling (4-connectivity) over the cleaned
   palette-index map.
6. Merge regions smaller than `--min-region` into the adjacent region with
   which they share the longest boundary. Then, if `--max-regions` is set,
   keep merging the smallest regions until the count is at most that target.
7. Render three outputs:
   - `preview.png` — what the finished painting should look like.
   - `template.png` — white canvas with black region outlines and the
     palette-index digit at each region's safest anchor (furthest pixel
     from every boundary). Digit size adapts to region size; tiny regions
     get a lead-line to a digit drawn outside the region. Outlines are
     dilated after upscale so edges stay crisp when printed.
   - `legend.png` — numbered colour swatches with RGB labels.
8. Dump the palette to `palette.json`.

## Install

```
pip install -e .
```

Dependencies (from `pyproject.toml`): `numpy`, `Pillow`, `scikit-image`,
`scikit-learn`, `scipy`. The optional `opencv` extra (`pip install -e
'.[opencv]'`) installs `opencv-contrib-python`, which enables
`--smooth meanshift` and `--saliency auto`'s OpenCV saliency backend
(otherwise `--saliency auto` falls back to a Sobel-magnitude weight map).

## CLI

```
python -m pbn input.jpg -o out/ \
  -k 12 \
  --smooth bilateral \
  --cleanup majority \
  --max-regions 400 \
  --min-region 30 \
  --min-delta-e 7.0 \
  --scale 6
```

Options:

- `-k / --colors` — number of palette colours (default 12). The *effective*
  count after delta-E collapse is reported in `palette.json` as
  `effective_k`.
- `--smooth {none,gaussian,bilateral,meanshift}` — pre-quantisation
  smoothing (default `gaussian`). `bilateral` and `meanshift` preserve
  edges while flattening textured regions; `gaussian` uses `--blur` as the
  sigma.
- `--blur` — Gaussian sigma when `--smooth gaussian` (default 0).
- `--cleanup {none,majority}` — label-map cleanup after quantisation
  (default `majority`).
- `--min-region` — pixel count below which a region is absorbed into its
  longest-shared-border neighbour (default 20, `0` disables).
- `--max-regions` — cap the number of 4-connected regions by iteratively
  merging the smallest ones. Disabled by default.
- `--max-per-color` — cap the number of 4-connected components per
  individual palette colour. Smallest fragments of an over-budget colour
  are repainted to their longest-bordered different-colour neighbour, so
  textured areas stop spawning scattered same-numbered specks. Disabled by
  default.
- `--min-delta-e` — minimum CIE76 Lab distance between any two palette
  centroids (default 7.0, `0` disables).
- `--saliency {none,center,auto}` — per-pixel `sample_weight` for K-means.
  `none` (default) keeps the unweighted fit; `center` uses a Gaussian
  centred on the canvas (cheap, no extra deps); `auto` calls
  `cv2.saliency.StaticSaliencyFineGrained` when `opencv-contrib-python` is
  installed, with a Sobel-magnitude fallback otherwise.
- `--scale` — nearest-neighbour upscale of the template so digits are
  legible (default 4). Outlines are dilated 1 px at scale ≥ 4, 2 px at
  scale ≥ 6.
- `--seed` — K-means random state for reproducible palettes.

## Output

- `preview.png` — `(H, W, 3)` colour preview.
- `template.png` — `(H·scale, W·scale, 3)` printable template.
- `legend.png` — palette legend.
- `palette.json` — `{"k": N, "effective_k": M, "min_delta_e": F,
  "saliency": "...", "colors": [{"index": 1, "rgb": [r, g, b]}, …]}`.

## Programmatic use

```python
from pbn.io import load_image
from pbn.pipeline import generate

image = load_image("photo.jpg")
result = generate(
    image,
    k=12,
    min_region_size=30,
    blur_sigma=0.0,
    smooth="bilateral",
    cleanup="majority",
    max_regions=400,
    min_delta_e=7.0,
    template_scale=6,
)

# result.palette      (k, 3) uint8 — first effective_k rows are the active palette
# result.effective_k  int — colours surviving delta-E collapse
# result.indices      (H, W) int32 — palette index per pixel
# result.preview      (H, W, 3) uint8
# result.template     (H*scale, W*scale, 3) uint8
# result.legend       (?, ?, 3) uint8
```

## Tests

```
pytest
```

85+ tests across IO, Lab quantisation, region labelling and merging,
boundary detection, label placement (including lead-lines for tiny
regions), template rendering (including dilated outlines), saliency
weighting, the full pipeline, and the CLI (in-process + subprocess smoke).
