# paint-by-number

Turn a photograph into a paint-by-number kit: a colour preview, a printable
template with numbered regions, and a palette legend.

## Pipeline

1. Load the input image (any format Pillow can read — PNG, JPEG, …).
2. Optional Gaussian blur to suppress noise.
3. K-means colour quantisation to `k` palette colours.
4. Connected-component labelling (4-connectivity) over the palette-index map.
5. Merge regions smaller than `--min-region` into the adjacent region with
   which they share the longest boundary.
6. Render three outputs:
   - `preview.png` — what the finished painting should look like.
   - `template.png` — white canvas with black region outlines and the
     palette-index digit placed at each region's safest anchor (the pixel
     furthest from every boundary, via distance transform).
   - `legend.png` — numbered colour swatches with RGB labels.
7. Dump the palette to `palette.json`.

## Install

```
pip install numpy Pillow scikit-learn scipy
```

## CLI

```
python -m pbn input.jpg -o out/ -k 12 --min-region 30 --blur 1.0 --scale 6
```

Options:

- `-k / --colors` — number of palette colours (default 12).
- `--min-region` — pixel count below which a region is absorbed into its
  neighbour (default 20, `0` disables).
- `--blur` — Gaussian blur sigma pre-quantisation (default 0).
- `--scale` — nearest-neighbour upscale of the template so digits are
  legible (default 4).
- `--seed` — K-means random state for reproducible palettes.

## Output

- `preview.png` — `(H, W, 3)` colour preview.
- `template.png` — `(H·scale, W·scale, 3)` printable template.
- `legend.png` — palette legend.
- `palette.json` — `{"k": N, "colors": [{"index": 1, "rgb": [r, g, b]}, …]}`.

## Programmatic use

```python
from pbn.io import load_image
from pbn.pipeline import generate

image = load_image("photo.jpg")
result = generate(image, k=10, min_region_size=25, blur_sigma=1.0, template_scale=5)

# result.palette  (k, 3) uint8
# result.indices  (H, W) int32
# result.preview  (H, W, 3) uint8
# result.template (H*5, W*5, 3) uint8
# result.legend   (?, ?, 3) uint8
```

## Tests

```
pytest
```

Implemented TDD — 46 tests across IO, quantisation, region labelling,
boundary detection, label placement, rendering, the full pipeline, and the
CLI (in-process + subprocess smoke).
