"""
Train EfficientNetB4UNet (models/efficientnet_unet.py) on
Dataset001_BrainTumorProcessed.

Matches the described training regimen:
  - AdamW, lr=1e-4, betas=(0.9, 0.999), eps=1e-8
  - ReduceLROnPlateau on val loss, factor=0.1
  - EarlyStopping on val loss, patience=10 epochs
  - 100 epochs, batch size 4
  - 90/10 train/test split
  - Dice loss
  - Metrics: accuracy, Dice, recall, precision, mean IoU, IoU

Usage:
  python train_efficientnet_unet.py
  python train_efficientnet_unet.py --epochs 100 --batch-size 4 --output-dir training_output/efficientnet_unet
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from models.efficientnet_unet import EfficientNetB4UNet

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class BrainTumorDataset(Dataset):
    """Reads Dataset001_BrainTumorProcessed nnUNet-raw PNGs.

    imagesTr/<case>_0000.png -> single-channel MRI slice, uint8 [0, 255]
    labelsTr/<case>.png      -> single-channel binary mask, uint8 {0, 1}
    """

    def __init__(self, images_dir: Path, labels_dir: Path, case_ids: list[str]):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.case_ids = case_ids

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        from PIL import Image

        image = np.array(Image.open(self.images_dir / f"{case_id}_0000.png"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.labels_dir / f"{case_id}.png"), dtype=np.float32)

        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)     # (1, H, W)
        return image, mask


def build_datasets(raw_dir: Path, val_fraction: float, seed: int):
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"
    case_ids = sorted(p.stem[:-5] for p in images_dir.glob("*_0000.png"))
    if not case_ids:
        raise RuntimeError(f"No cases found under {images_dir}")

    full_dataset = BrainTumorDataset(images_dir, labels_dir, case_ids)
    n_val = max(1, round(len(case_ids) * val_fraction))
    n_train = len(case_ids) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)
    return train_set, val_set


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss on sigmoid probabilities (model already applies sigmoid)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs, targets):
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(probs, targets, threshold=0.5, eps=1e-6):
    """probs, targets: (N, 1, H, W). Returns a dict of batch-summed stats."""
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou_fg = (tp + eps) / (tp + fp + fn + eps)
    iou_bg = (tn + eps) / (tn + fp + fn + eps)
    mean_iou = (iou_fg + iou_bg) / 2
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "accuracy": accuracy,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "iou": iou_fg,
        "mean_iou": mean_iou,
    }


def average_metrics(metric_dicts):
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}


# --------------------------------------------------------------------------
# Train / validate
# --------------------------------------------------------------------------

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_samples = 0
    batch_metrics = []

    torch.set_grad_enabled(is_train)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        if is_train:
            optimizer.zero_grad()

        probs = model(images)
        loss = criterion(probs, masks)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        batch_metrics.append(compute_metrics(probs.detach(), masks))

    torch.set_grad_enabled(True)
    return total_loss / n_samples, average_metrics(batch_metrics)


def get_device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nnunet-raw", default=str(ROOT / "nnUNet_raw" / "Dataset001_BrainTumorProcessed"))
    parser.add_argument("--output-dir", default=str(ROOT / "training_output" / "efficientnet_unet"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="ReduceLROnPlateau patience (epochs of no val-loss improvement before lr *= factor)")
    parser.add_argument("--lr-factor", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    raw_dir = Path(args.nnunet_raw)
    train_set, val_set = build_datasets(raw_dir, args.val_fraction, args.seed)
    print(f"Train: {len(train_set)} cases | Val: {len(val_set)} cases")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = EfficientNetB4UNet(n_channels=1, n_classes=1, pretrained=args.pretrained).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    log_path = out_dir / "training_log.csv"
    fieldnames = ["epoch", "lr", "train_loss", "val_loss",
                  "val_accuracy", "val_dice", "val_precision", "val_recall", "val_iou", "val_mean_iou",
                  "epoch_time_s"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        row = {
            "epoch": epoch, "lr": current_lr, "train_loss": train_loss, "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"], "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
            "val_iou": val_metrics["iou"], "val_mean_iou": val_metrics["mean_iou"],
            "epoch_time_s": epoch_time,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(f"[{epoch:3d}/{args.epochs}] lr={current_lr:.2e} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
              f"({epoch_time:.1f}s)")

        improved = val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
                "val_metrics": val_metrics,
            }, out_dir / "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping: no val_loss improvement for {args.early_stopping_patience} epochs.")
            break

    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
        "val_metrics": val_metrics,
    }, out_dir / "final_model.pth")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}. "
          f"Checkpoints + training_log.csv saved to {out_dir}")


if __name__ == "__main__":
    main()
