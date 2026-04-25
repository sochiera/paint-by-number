from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pbn.io import load_image, save_image
from pbn.pipeline import CLEANUP_CHOICES, SMOOTHING_CHOICES, generate
from pbn.print_size import PRINT_SIZES, resolve_print_params
from pbn.saliency import SALIENCY_MODES
from pbn.segment import PRESEGMENT_CHOICES


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pbn",
        description="Turn a photograph into a paint-by-number template.",
    )
    p.add_argument("input", type=Path, help="input image (PNG, JPEG, ...).")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output directory (created if it does not exist).",
    )
    p.add_argument(
        "-k",
        "--colors",
        type=int,
        default=12,
        help="number of palette colours (default: 12).",
    )
    p.add_argument(
        "--min-region",
        type=int,
        default=None,
        dest="min_region",
        help=(
            "merge connected regions smaller than this (source pixels). "
            "0 disables. When --print-size is set, defaults to the smallest "
            "region that still covers ~4 mm² on the printed output; "
            "otherwise defaults to 20."
        ),
    )
    p.add_argument(
        "--blur",
        type=float,
        default=0.0,
        help=(
            "Gaussian blur sigma applied before quantisation (0 disables). "
            "Kept for backwards compatibility; prefer --smooth."
        ),
    )
    p.add_argument(
        "--smooth",
        choices=SMOOTHING_CHOICES,
        default="gaussian",
        help=(
            "pre-quantisation smoothing filter. 'gaussian' (default) uses "
            "--blur as sigma; 'bilateral' and 'meanshift' preserve edges "
            "while flattening textured regions; 'none' skips smoothing."
        ),
    )
    p.add_argument(
        "--cleanup",
        choices=CLEANUP_CHOICES,
        default="majority",
        help=(
            "label-map cleanup applied after quantisation. 'majority' "
            "(default) runs a 3x3 majority filter that dissolves isolated "
            "speckles without moving strong contours; 'none' disables it."
        ),
    )
    p.add_argument(
        "--max-regions",
        type=int,
        default=None,
        dest="max_regions",
        help=(
            "cap the number of 4-connected regions in the output by "
            "iteratively merging the smallest regions into their longest-"
            "border neighbour. Disabled by default."
        ),
    )
    p.add_argument(
        "--max-per-color",
        type=int,
        default=None,
        dest="max_per_color",
        help=(
            "cap the number of 4-connected components per palette colour "
            "by repainting the smallest over-budget fragments to their "
            "longest-bordered different-colour neighbour. Suppresses "
            "scattered same-numbered specks in textured areas. Disabled "
            "by default."
        ),
    )
    p.add_argument(
        "--min-delta-e",
        type=float,
        default=7.0,
        dest="min_delta_e",
        help=(
            "minimum CIE76 Lab distance between any two palette centroids. "
            "Pairs closer than this are collapsed by re-running K-means "
            "with k-1 until all remaining centroids are this far apart "
            "(or only two remain). 0 disables. Default 7.0."
        ),
    )
    p.add_argument(
        "--presegment",
        choices=PRESEGMENT_CHOICES,
        default="none",
        help=(
            "pre-quantisation segmentation. 'none' (default) feeds the "
            "smoothed image straight to K-means; 'slic' runs SLIC and "
            "replaces every pixel with its superpixel's mean RGB, which "
            "collapses textured regions before quantisation."
        ),
    )
    p.add_argument(
        "--slic-segments",
        type=int,
        default=600,
        dest="slic_segments",
        help=(
            "approximate number of SLIC superpixels when "
            "--presegment slic. Higher = more detail (default: 600)."
        ),
    )
    p.add_argument(
        "--slic-compactness",
        type=float,
        default=10.0,
        dest="slic_compactness",
        help=(
            "SLIC spatial-vs-colour trade-off when --presegment slic. "
            "Larger values give more compact, square-ish superpixels "
            "(default: 10.0)."
        ),
    )
    p.add_argument(
        "--saliency",
        choices=SALIENCY_MODES,
        default="none",
        help=(
            "per-pixel weighting passed as sample_weight to K-means. "
            "'none' (default) keeps the unweighted fit; 'center' biases "
            "centroids towards the canvas centre; 'auto' uses "
            "cv2.saliency.StaticSaliencyFineGrained when opencv-contrib-"
            "python is installed, otherwise a Sobel-magnitude fallback."
        ),
    )
    p.add_argument(
        "--scale",
        type=int,
        default=None,
        help=(
            "template upscale factor for readable numbers. When "
            "--print-size is set, defaults to the largest integer scale "
            "that fits the page; otherwise defaults to 4."
        ),
    )
    p.add_argument(
        "--print-size",
        choices=PRINT_SIZES,
        default=None,
        dest="print_size",
        help=(
            "target print page size; combined with --dpi, derives "
            "sensible defaults for --scale and --min-region. Page "
            "orientation matches the image's aspect."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help=(
            "target print resolution in dots per inch (default: 300). "
            "Only used when --print-size is set."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for K-means (default: 0).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.colors < 1:
        parser.error(f"--colors must be >= 1, got {args.colors}")
    if args.dpi < 1:
        parser.error(f"--dpi must be >= 1, got {args.dpi}")
    if args.scale is not None and args.scale < 1:
        parser.error(f"--scale must be >= 1, got {args.scale}")
    if args.max_regions is not None and args.max_regions < 1:
        parser.error(
            f"--max-regions must be >= 1, got {args.max_regions}"
        )
    if args.max_per_color is not None and args.max_per_color < 1:
        parser.error(
            f"--max-per-color must be >= 1, got {args.max_per_color}"
        )
    if args.slic_segments < 1:
        parser.error(
            f"--slic-segments must be >= 1, got {args.slic_segments}"
        )
    if args.slic_compactness <= 0:
        parser.error(
            f"--slic-compactness must be > 0, got {args.slic_compactness}"
        )
    if args.min_delta_e < 0:
        parser.error(
            f"--min-delta-e must be >= 0, got {args.min_delta_e}"
        )

    image = load_image(args.input)

    # Resolve print-size defaults *after* the image is loaded so we know its
    # aspect ratio. Explicit --scale / --min-region always win.
    if args.print_size is not None:
        resolution = resolve_print_params(
            args.print_size,
            dpi=args.dpi,
            image_h=image.shape[0],
            image_w=image.shape[1],
        )
        if args.scale is None:
            args.scale = resolution.scale
        if args.min_region is None:
            args.min_region = resolution.min_region_size
    if args.scale is None:
        args.scale = 4
    if args.min_region is None:
        args.min_region = 20

    result = generate(
        image,
        k=args.colors,
        min_region_size=args.min_region,
        blur_sigma=args.blur,
        template_scale=args.scale,
        random_state=args.seed,
        smooth=args.smooth,
        max_regions=args.max_regions,
        max_per_color=args.max_per_color,
        cleanup=args.cleanup,
        min_delta_e=args.min_delta_e,
        saliency=args.saliency,
        presegment=args.presegment,
        slic_segments=args.slic_segments,
        slic_compactness=args.slic_compactness,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_image(result.preview, args.output / "preview.png")
    save_image(result.template, args.output / "template.png")
    save_image(result.legend, args.output / "legend.png")

    palette_json = {
        "k": int(len(result.palette)),
        "effective_k": int(result.effective_k),
        "min_delta_e": float(args.min_delta_e),
        "saliency": args.saliency,
        "colors": [
            {"index": i + 1, "rgb": [int(c) for c in rgb]}
            for i, rgb in enumerate(result.palette[: result.effective_k])
        ],
    }
    (args.output / "palette.json").write_text(json.dumps(palette_json, indent=2))

    print(f"Wrote {args.output}/{{preview,template,legend}}.png + palette.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
