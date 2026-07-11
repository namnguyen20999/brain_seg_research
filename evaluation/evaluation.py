# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# # Brain Tumor Segmentation — Model Evaluation
# **Model:** nnUNet 2D (PlainConvUNet, 8 stages)  
# **Dataset:** Brain Tumor MRI Segmentation (binary: background / tumor)  

# %%
import json, os, re, zipfile, io, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# %%
# Paths
NNUNET_DIR   = '/content/drive/MyDrive/nnUNet_results'
RESULTS_DIR  = os.path.join(NNUNET_DIR, 'Dataset001_BrainTumor/nnUNetTrainer_50epochs__nnUNetPlans__2d')
BRAIN_SEG_DIR = '/content/drive/MyDrive/brain_segmentation_dataset/archive'
IMG_DIR      = os.path.join(BRAIN_SEG_DIR, 'images')
MASK_DIR     = os.path.join(BRAIN_SEG_DIR, 'masks')

FOLDS        = [f'fold_{i}' for i in range(5)]
VAL_DIRS     = {f: os.path.join(RESULTS_DIR, f, 'validation') for f in FOLDS}
SUMMARY_FILES = {f: os.path.join(RESULTS_DIR, f, 'validation', 'summary.json') for f in FOLDS}

for f in FOLDS:
    print(f'{f}:', all(os.path.exists(p) for p in [VAL_DIRS[f], SUMMARY_FILES[f]]))
print('images :', os.path.exists(IMG_DIR))
print('masks  :', os.path.exists(MASK_DIR))

# %%
# Dice & IoU Metrics

records = []
for f in FOLDS:
    with open(SUMMARY_FILES[f]) as fp:
        s = json.load(fp)
    for case in s['metric_per_case']:
        m = case['metrics']['1']
        records.append({
            'fold': f,
            'case': os.path.basename(case['reference_file']),
            'Dice': m['Dice'],
            'IoU':  m['IoU'],
        })

df = pd.DataFrame(records)

fold_summary = df.groupby('fold')[['Dice', 'IoU']].mean().round(4)
fold_summary.loc['Overall'] = df[['Dice', 'IoU']].mean().round(4)
fold_summary.loc['Std']     = df[['Dice', 'IoU']].std().round(4)
print(fold_summary.to_string())


# %%
# Build Case Map (5-fold)
img_files  = {Path(f).stem: os.path.join(IMG_DIR, f)  for f in os.listdir(IMG_DIR)}
mask_files = {Path(f).stem: os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR)}

case_map = {}
for f in FOLDS:
    for fn in os.listdir(VAL_DIRS[f]):
        if not fn.endswith('.png'):
            continue
        stem     = Path(fn).stem
        img_stem = stem[:-5] if stem.endswith('_0000') else stem
        img      = img_files.get(img_stem)
        mask     = mask_files.get(img_stem)
        if not img or not mask:
            continue
        dice_row = df[(df['fold'] == f) & (df['case'] == f'{stem}.png')]
        case_map[stem] = {
            'fold':  f,
            'pred':  os.path.join(VAL_DIRS[f], fn),
            'image': img,
            'mask':  mask,
            'dice':  dice_row['Dice'].values[0] if len(dice_row) else None,
            'iou':   dice_row['IoU'].values[0]  if len(dice_row) else None,
        }

total = sum(len([fn for fn in os.listdir(VAL_DIRS[f]) if fn.endswith('.png')]) for f in FOLDS)
print(f'Total predictions : {total}')
print(f'Matched           : {len(case_map)}')
print(f'With metrics      : {sum(1 for v in case_map.values() if v["dice"] is not None)}')


# %%
def plot_cases(cases, title):
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 4 * len(cases)))
    if len(cases) == 1:
        axes = [axes]

    col_titles = ['MRI Image', 'Ground Truth (red)', 'Prediction (green)',
                  'GT vs Pred\n(red=FN  green=FP  yellow=TP)']
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, fontweight='bold')

    for row, c in enumerate(cases):
        img      = np.array(Image.open(c['image']).convert('L'))
        gt_bin   = np.array(Image.open(c['mask']).convert('L')) > 0
        pred_bin = np.array(Image.open(c['pred']).convert('L')) > 0

        dice_str = f"Dice={c['dice']:.3f}  IoU={c['iou']:.3f}" if c['dice'] is not None else 'no metrics'
        axes[row][0].imshow(img, cmap='gray')
        axes[row][0].set_ylabel(f"{c['fold']}\n{dice_str}", fontsize=8)

        gt_overlay = np.zeros((*img.shape, 4), dtype=float)
        gt_overlay[gt_bin] = [1, 0, 0, 0.45]        # red
        axes[row][1].imshow(img, cmap='gray')
        axes[row][1].imshow(gt_overlay)

        pred_overlay = np.zeros((*img.shape, 4), dtype=float)
        pred_overlay[pred_bin] = [0, 1, 0, 0.45]    # green
        axes[row][2].imshow(img, cmap='gray')
        axes[row][2].imshow(pred_overlay)

        combined = np.zeros((*img.shape, 4), dtype=float)
        combined[gt_bin & ~pred_bin]  = [1, 0, 0, 0.5]   # red    — missed (FN)
        combined[pred_bin & ~gt_bin]  = [0, 1, 0, 0.5]   # green  — false pos (FP)
        combined[gt_bin & pred_bin]   = [1, 1, 0, 0.6]   # yellow — correct (TP)
        axes[row][3].imshow(img, cmap='gray')
        axes[row][3].imshow(combined)

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.002)
    plt.tight_layout()
    plt.show()


# %%
samples = random.sample(list(case_map.values()), min(5, len(case_map)))
plot_cases(samples, 'nnUNet 2D — Random Sample (5 cases)')

# %%
# ── Worst 30 predictions (Dice below threshold) ───────────────────────────────
THRESHOLD = 0.3

worst_30 = sorted(
    [v for v in case_map.values() if v['dice'] is not None and v['dice'] < THRESHOLD],
    key=lambda x: x['dice']
)[:30]

print(f'Showing worst {len(worst_30)} cases (Dice < {THRESHOLD}):')
for i, c in enumerate(worst_30):
    print(f"  #{i+1:2d}  {c['fold']}  Dice={c['dice']:.4f}  IoU={c['iou']:.4f}  {Path(c['pred']).stem}")

fig, axes = plt.subplots(len(worst_30), 4, figsize=(16, 4 * len(worst_30)))
col_titles = ['MRI Image', 'Ground Truth (red)', 'Prediction (green)',
              'GT vs Pred\n(red=FN  green=FP  yellow=TP)']
for ax, t in zip(axes[0], col_titles):
    ax.set_title(t, fontsize=9, fontweight='bold')

for row, c in enumerate(worst_30):
    img      = np.array(Image.open(c['image']).convert('L'))
    gt_bin   = np.array(Image.open(c['mask']).convert('L')) > 0
    pred_bin = np.array(Image.open(c['pred']).convert('L')) > 0

    img_id   = Path(c['image']).stem
    dice_str = f"Dice={c['dice']:.3f}  IoU={c['iou']:.3f}"
    axes[row][0].imshow(img, cmap='gray')
    axes[row][0].set_ylabel(f"{img_id}\n{c['fold']}  {dice_str}", fontsize=8)
    axes[row][0].text(4, 14, img_id, color='yellow', fontsize=9, fontweight='bold')

    gt_overlay = np.zeros((*img.shape, 4), dtype=float)
    gt_overlay[gt_bin] = [1, 0, 0, 0.45]
    axes[row][1].imshow(img, cmap='gray')
    axes[row][1].imshow(gt_overlay)

    pred_overlay = np.zeros((*img.shape, 4), dtype=float)
    pred_overlay[pred_bin] = [0, 1, 0, 0.45]
    axes[row][2].imshow(img, cmap='gray')
    axes[row][2].imshow(pred_overlay)

    combined = np.zeros((*img.shape, 4), dtype=float)
    combined[gt_bin & ~pred_bin] = [1, 0, 0, 0.5]
    combined[pred_bin & ~gt_bin] = [0, 1, 0, 0.5]
    combined[gt_bin & pred_bin]  = [1, 1, 0, 0.6]
    axes[row][3].imshow(img, cmap='gray')
    axes[row][3].imshow(combined)

    for ax in axes[row]:
        ax.axis('off')

plt.suptitle(f'nnUNet 2D — Worst {len(worst_30)} Predictions (Dice < {THRESHOLD})',
             fontsize=13, fontweight='bold', y=1.002)
plt.tight_layout()
plt.show()


# %%



