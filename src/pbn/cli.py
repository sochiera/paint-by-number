from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pbn.io import load_image, save_image
from pbn.pipeline import generate


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
        help="Gaussian blur sigma applied before quantisation (0 disables).",
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

    image = load_image(args.input)
    result = generate(
        image,
        k=args.colors,
        min_region_size=args.min_region,
        blur_sigma=args.blur,
        template_scale=args.scale,
        random_state=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_image(result.preview, args.output / "preview.png")
    save_image(result.template, args.output / "template.png")
    save_image(result.legend, args.output / "legend.png")

    palette_json = {
        "k": int(len(result.palette)),
        "colors": [
            {"index": i + 1, "rgb": [int(c) for c in rgb]}
            for i, rgb in enumerate(result.palette)
        ],
    }
    (args.output / "palette.json").write_text(json.dumps(palette_json, indent=2))

    print(f"Wrote {args.output}/{{preview,template,legend}}.png + palette.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
