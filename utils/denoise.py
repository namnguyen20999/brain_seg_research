"""
Denoise grayscale MRI slices (e.g. 2D_dataset_T1_png) with a choice of filters.

Usage:
    python3 utils/denoise.py --input path/to/image.png --output path/to/out.png
    python3 utils/denoise.py --input path/to/dir --output path/to/out_dir --method nlm
    python3 utils/denoise.py --input path/to/dir --output path/to/out_dir --method gaussian --sigma 1.0
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.restoration import denoise_nl_means, denoise_bilateral, estimate_sigma
from scipy.ndimage import gaussian_filter, median_filter

METHODS = ("nlm", "gaussian", "median", "bilateral")


def denoise(image: np.ndarray, method: str = "nlm", sigma: float = 1.0, size: int = 3) -> np.ndarray:
    """Denoise a single-channel image (any numeric dtype, returned as float64 in [0, 1])."""
    arr = image.astype(np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0

    if method == "nlm":
        noise_std = estimate_sigma(arr)
        return denoise_nl_means(arr, h=1.15 * noise_std, fast_mode=True, patch_size=5, patch_distance=6)
    elif method == "gaussian":
        return gaussian_filter(arr, sigma=sigma)
    elif method == "median":
        return median_filter(arr, size=size)
    elif method == "bilateral":
        return denoise_bilateral(arr, sigma_color=0.05, sigma_spatial=sigma)
    else:
        raise ValueError(f"Unknown method '{method}', expected one of {METHODS}")


def denoise_file(input_path: Path, output_path: Path, method: str, sigma: float, size: int) -> None:
    image = np.array(Image.open(input_path).convert("L"))
    result = denoise(image, method=method, sigma=sigma, size=size)
    result_uint8 = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result_uint8).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Denoise MRI slice PNGs")
    parser.add_argument("--input", required=True, type=Path, help="Input image file or directory")
    parser.add_argument("--output", required=True, type=Path, help="Output image file or directory")
    parser.add_argument("--method", choices=METHODS, default="nlm", help="Denoising method")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for gaussian/bilateral")
    parser.add_argument("--size", type=int, default=3, help="Filter window size for median")
    args = parser.parse_args()

    if args.input.is_dir():
        paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for path in paths:
            out_path = args.output / path.relative_to(args.input)
            denoise_file(path, out_path, args.method, args.sigma, args.size)
        print(f"Denoised {len(paths)} images -> {args.output}")
    else:
        denoise_file(args.input, args.output, args.method, args.sigma, args.size)
        print(f"Denoised {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
