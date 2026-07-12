"""
Run the full preprocessing pipeline in order: denoise -> resize -> CLAHE.

Each image is processed in memory and written once, so the output is a
single preprocessed dataset (no intermediate per-stage folders, no
visualization output).

Expects a dataset laid out as:
    <input>/<patient_id>/images/<name>.png
    <input>/<patient_id>/masks/<name>.png

Writes:
    <output>/<patient_id>/images/<name>.png   — denoised, resized, CLAHE'd
    <output>/<patient_id>/masks/<name>.png    — resized only (nearest-neighbor)

Usage:
    python3 utils/run_pipeline.py --input datasets/2D_dataset_T1_png --output datasets/pipeline_output --size 256 256
    python3 utils/run_pipeline.py --input datasets/2D_dataset_T1_png --output datasets/pipeline_output --size 256 256 \
        --denoise-method nlm --clip-limit 0.01 --tile-size 8
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from denoise import denoise
from clahe import apply_clahe
from resize import DOWNSCALE_RESAMPLE, MASK_RESAMPLE


def process_image(img_path: Path, out_path: Path, size: tuple[int, int], denoise_method: str,
                   sigma: float, median_size: int, clip_limit: float, tile_size: int) -> None:
    image = np.array(Image.open(img_path).convert("L"))

    denoised = denoise(image, method=denoise_method, sigma=sigma, size=median_size)
    denoised_uint8 = (np.clip(denoised, 0.0, 1.0) * 255).astype(np.uint8)

    resized = Image.fromarray(denoised_uint8).resize(size, resample=DOWNSCALE_RESAMPLE)

    clahed = apply_clahe(np.array(resized), clip_limit=clip_limit, tile_size=tile_size)
    clahed_uint8 = (np.clip(clahed, 0.0, 1.0) * 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(clahed_uint8).save(out_path)


def process_mask(mask_path: Path, out_path: Path, size: tuple[int, int]) -> None:
    mask = Image.open(mask_path).resize(size, resample=MASK_RESAMPLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run denoise -> resize -> CLAHE in order, writing one final dataset")
    parser.add_argument("--input", required=True, type=Path, help="Dataset root with <patient>/images and <patient>/masks")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for the preprocessed dataset")
    parser.add_argument("--size", required=True, type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Target resize dimensions")
    parser.add_argument("--denoise-method", choices=("nlm", "gaussian", "median", "bilateral"), default="nlm")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for gaussian/bilateral denoising")
    parser.add_argument("--median-size", type=int, default=3, help="Filter window size for median denoising")
    parser.add_argument("--clip-limit", type=float, default=0.01, help="CLAHE contrast clip limit")
    parser.add_argument("--tile-size", type=int, default=8, help="CLAHE local tile size")
    args = parser.parse_args()

    size = tuple(args.size)
    patient_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())

    img_count = 0
    mask_count = 0
    for patient_dir in patient_dirs:
        images_dir = patient_dir / "images"
        masks_dir = patient_dir / "masks"

        if images_dir.is_dir():
            for img_path in sorted(images_dir.glob("*.png")):
                out_path = args.output / patient_dir.name / "images" / img_path.name
                process_image(img_path, out_path, size, args.denoise_method, args.sigma,
                              args.median_size, args.clip_limit, args.tile_size)
                img_count += 1

        if masks_dir.is_dir():
            for mask_path in sorted(masks_dir.glob("*.png")):
                out_path = args.output / patient_dir.name / "masks" / mask_path.name
                process_mask(mask_path, out_path, size)
                mask_count += 1

    print(f"Processed {img_count} images, {mask_count} masks -> {args.output}")


if __name__ == "__main__":
    main()
