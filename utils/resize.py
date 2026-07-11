"""
Scale down (or up) images to a target size.

Usage:
    python3 utils/resize.py --input path/to/image.png --output path/to/out.png --size 256 256
    python3 utils/resize.py --input path/to/dir --output path/to/out_dir --size 256 256
    python3 utils/resize.py --input path/to/dir --output path/to/out_dir --size 256 256 --mask
"""

import argparse
from pathlib import Path

from PIL import Image

DOWNSCALE_RESAMPLE = Image.BILINEAR
MASK_RESAMPLE = Image.NEAREST


def resize_file(input_path: Path, output_path: Path, size: tuple[int, int], is_mask: bool) -> None:
    image = Image.open(input_path)
    resample = MASK_RESAMPLE if is_mask else DOWNSCALE_RESAMPLE
    resized = image.resize(size, resample=resample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Resize images to a target dimension")
    parser.add_argument("--input", required=True, type=Path, help="Input image file or directory")
    parser.add_argument("--output", required=True, type=Path, help="Output image file or directory")
    parser.add_argument("--size", required=True, type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Target size")
    parser.add_argument("--mask", action="store_true", help="Use nearest-neighbor resampling (for label/mask images)")
    args = parser.parse_args()

    size = tuple(args.size)

    if args.input.is_dir():
        paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for path in paths:
            out_path = args.output / path.relative_to(args.input)
            resize_file(path, out_path, size, args.mask)
        print(f"Resized {len(paths)} images to {size} -> {args.output}")
    else:
        resize_file(args.input, args.output, size, args.mask)
        print(f"Resized {args.input} to {size} -> {args.output}")


if __name__ == "__main__":
    main()
