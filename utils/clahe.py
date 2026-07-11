"""
Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to grayscale MRI slices.

Usage:
    python3 utils/clahe.py --input path/to/image.png --output path/to/out.png
    python3 utils/clahe.py --input path/to/dir --output path/to/out_dir
    python3 utils/clahe.py --input path/to/dir --output path/to/out_dir --clip-limit 0.02 --tile-size 8
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist


def apply_clahe(image: np.ndarray, clip_limit: float = 0.01, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE to a single-channel image (any numeric dtype, returned as float64 in [0, 1])."""
    arr = image.astype(np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return equalize_adapthist(arr, kernel_size=tile_size, clip_limit=clip_limit)


def clahe_file(input_path: Path, output_path: Path, clip_limit: float, tile_size: int) -> None:
    image = np.array(Image.open(input_path).convert("L"))
    result = apply_clahe(image, clip_limit=clip_limit, tile_size=tile_size)
    result_uint8 = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result_uint8).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Apply CLAHE to MRI slice PNGs")
    parser.add_argument("--input", required=True, type=Path, help="Input image file or directory")
    parser.add_argument("--output", required=True, type=Path, help="Output image file or directory")
    parser.add_argument("--clip-limit", type=float, default=0.01, help="Contrast clip limit (higher = more contrast)")
    parser.add_argument("--tile-size", type=int, default=8, help="Size of the local tiles (kernel_size)")
    args = parser.parse_args()

    if args.input.is_dir():
        paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for path in paths:
            out_path = args.output / path.relative_to(args.input)
            clahe_file(path, out_path, args.clip_limit, args.tile_size)
        print(f"Applied CLAHE to {len(paths)} images -> {args.output}")
    else:
        clahe_file(args.input, args.output, args.clip_limit, args.tile_size)
        print(f"Applied CLAHE to {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
