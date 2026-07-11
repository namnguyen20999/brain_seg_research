"""
Overlay segmentation masks on their images for visual inspection.

Expects a dataset laid out as:
    <input>/<patient_id>/images/<name>.png
    <input>/<patient_id>/masks/<name>.png

Writes:
    <output>/<patient_id>/<name>.png   — image with the mask drawn as a
                                          semi-transparent red overlay

Usage:
    python3 utils/visualize_overlay.py --input datasets/clahe_resized_256_2D_dataset_T1_png --output visualizations/clahe_resized_256_2D_dataset_T1_png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

OVERLAY_COLOR = (255, 0, 0)
OVERLAY_ALPHA = 0.4


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a binary mask onto a grayscale image as a semi-transparent color overlay."""
    rgb = np.stack([image] * 3, axis=-1).astype(np.float64)
    mask_bool = mask > 0
    color = np.array(OVERLAY_COLOR, dtype=np.float64)
    rgb[mask_bool] = (1 - OVERLAY_ALPHA) * rgb[mask_bool] + OVERLAY_ALPHA * color
    return rgb.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Overlay masks on images for visualization")
    parser.add_argument("--input", required=True, type=Path, help="Dataset root with <patient>/images and <patient>/masks")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for overlay images")
    args = parser.parse_args()

    patient_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    count = 0
    for patient_dir in patient_dirs:
        images_dir = patient_dir / "images"
        masks_dir = patient_dir / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            continue
        out_dir = args.output / patient_dir.name
        for img_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / img_path.name
            if not mask_path.is_file():
                continue
            image = np.array(Image.open(img_path).convert("L"))
            mask = np.array(Image.open(mask_path).convert("L"))
            overlaid = overlay_mask(image, mask)
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlaid).save(out_dir / img_path.name)
            count += 1

    print(f"Wrote {count} overlay images -> {args.output}")


if __name__ == "__main__":
    main()
