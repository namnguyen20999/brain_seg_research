"""
Place pipeline-processed images/masks into an nnUNet_raw dataset folder.

Expects a pipeline stage laid out as:
    <input>/<patient_id>/images/<slice_id>.png
    <input>/<patient_id>/masks/<slice_id>.png

Writes an nnUNet raw dataset (matches the existing Dataset001_BrainTumor layout):
    <nnunet-raw>/DatasetXXX_<name>/imagesTr/<slice_id>_0000.png
    <nnunet-raw>/DatasetXXX_<name>/labelsTr/<slice_id>.png   (values remapped 255 -> 1)
    <nnunet-raw>/DatasetXXX_<name>/dataset.json

Usage:
    python3 utils/prepare_nnunet_raw.py --input datasets/pipeline_output/03_clahe \
        --nnunet-raw nnUNet_raw --dataset-id 2 --dataset-name BrainTumorProcessed
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build_dataset(input_dir: Path, dataset_dir: Path, dataset_name: str) -> int:
    images_out = dataset_dir / "imagesTr"
    labels_out = dataset_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "imagesTs").mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    count = 0
    for patient_dir in patient_dirs:
        images_dir = patient_dir / "images"
        masks_dir = patient_dir / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            continue
        for img_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / img_path.name
            if not mask_path.is_file():
                continue
            slice_id = img_path.stem

            image = Image.open(img_path).convert("L")
            image.save(images_out / f"{slice_id}_0000.png")

            mask = np.array(Image.open(mask_path).convert("L"))
            mask_binary = (mask > 0).astype(np.uint8)
            Image.fromarray(mask_binary).save(labels_out / f"{slice_id}.png")

            count += 1

    return count


def write_dataset_json(dataset_dir: Path, dataset_name: str, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": num_training,
        "numTest": 0,
        "file_ending": ".png",
        "name": dataset_name,
        "description": "Brain tumor MRI segmentation (binary: background / tumor).",
    }
    with open(dataset_dir / "dataset.json", "w") as fp:
        json.dump(dataset_json, fp, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Build an nnUNet_raw dataset from a pipeline output stage")
    parser.add_argument("--input", required=True, type=Path, help="Pipeline stage dir with <patient>/images and <patient>/masks")
    parser.add_argument("--nnunet-raw", required=True, type=Path, help="nnUNet_raw root directory")
    parser.add_argument("--dataset-id", required=True, type=int, help="nnUNet dataset id, e.g. 2")
    parser.add_argument("--dataset-name", required=True, help="Dataset name suffix, e.g. BrainTumorProcessed")
    args = parser.parse_args()

    full_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = args.nnunet_raw / full_name

    count = build_dataset(args.input, dataset_dir, full_name)
    write_dataset_json(dataset_dir, full_name, count)

    print(f"Wrote {count} training cases -> {dataset_dir}")


if __name__ == "__main__":
    main()
