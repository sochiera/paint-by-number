from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pbn.io import load_image, save_image
from pbn.pipeline import CLEANUP_CHOICES, SMOOTHING_CHOICES, generate
from pbn.saliency import SALIENCY_MODES


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
        default=20,
        dest="min_region",
        help="merge connected regions smaller than this (pixels). 0 disables.",
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
        default=4,
        help="template upscale factor for readable numbers (default: 4).",
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
    if args.scale < 1:
        parser.error(f"--scale must be >= 1, got {args.scale}")
    if args.max_regions is not None and args.max_regions < 1:
        parser.error(
            f"--max-regions must be >= 1, got {args.max_regions}"
        )
    if args.min_delta_e < 0:
        parser.error(
            f"--min-delta-e must be >= 0, got {args.min_delta_e}"
        )

    image = load_image(args.input)
    result = generate(
        image,
        k=args.colors,
        min_region_size=args.min_region,
        blur_sigma=args.blur,
        template_scale=args.scale,
        random_state=args.seed,
        smooth=args.smooth,
        max_regions=args.max_regions,
        cleanup=args.cleanup,
        min_delta_e=args.min_delta_e,
        saliency=args.saliency,
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
