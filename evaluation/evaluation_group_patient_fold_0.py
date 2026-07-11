# %% [markdown]
# # Brain Tumor Segmentation — Model Evaluation
# **Model:** nnUNetTrainer_100epochs (2D), fold_0
# **Dataset:** Brain Tumor MRI Segmentation (binary: background / tumor)

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# %%
# Paths
BASE_DIR     = Path('/Users/namnguyen/PycharmProjects/brain_seg_research')
RESULTS_DIR  = BASE_DIR / 'nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer_100epochs__nnUNetPlans__2d'
PNG_DIR      = BASE_DIR / 'datasets/2D_dataset_T1_png'
NNUNET_RAW   = BASE_DIR / 'nnUNet_raw/Dataset001_BrainTumor'

# nnUNet fold_0 corresponds to splits.json fold_1 (cvind is 1-based)
FOLD         = 'fold_0'
CVIND_FOLD   = 'fold_1'
VAL_DIR      = RESULTS_DIR / FOLD / 'validation'
SUMMARY_FILE = VAL_DIR / 'summary.json'

print('val dir    :', VAL_DIR.exists())
print('summary    :', SUMMARY_FILE.exists())
print('nnunet raw :', NNUNET_RAW.exists())
print('splits.json:', (PNG_DIR / 'splits.json').exists())

# %%
# Build slice → patient map from 2D_dataset_T1_png directory tree
slice_to_patient = {}
for patient_dir in PNG_DIR.iterdir():
    if not patient_dir.is_dir():
        continue
    pid = patient_dir.name
    img_subdir = patient_dir / 'images'
    if img_subdir.exists():
        for img_path in img_subdir.glob('*.png'):
            slice_to_patient[img_path.stem] = pid  # stem = slice ID string

print(f'Slice→patient map: {len(slice_to_patient)} slices across {len(set(slice_to_patient.values()))} patients')

# Confirm val patients from splits.json
splits = json.load(open(PNG_DIR / 'splits.json'))
val_patients = set(splits[CVIND_FOLD]['val'])
print(f'Expected val patients ({CVIND_FOLD}): {len(val_patients)}')

# %%
# Dice & IoU Metrics (slice level)
records = []
with open(SUMMARY_FILE, encoding='utf-8') as fp:
    s = json.load(fp)

for case in s['metric_per_case']:
    m        = case['metrics']['1']
    slice_id = Path(case['reference_file']).stem   # e.g. "1003"
    patient  = slice_to_patient.get(slice_id, 'unknown')
    records.append({
        'patient': patient,
        'case':    Path(case['reference_file']).name,
        'Dice':    m['Dice'],
        'IoU':     m['IoU'],
    })

df = pd.DataFrame(records)

# ── Slice-level summary ────────────────────────────────────────────────────────
print('\nSlice-level summary:')
print(f"  n      : {len(df)} slices")
print(f"  Dice   : {df['Dice'].mean():.4f} ± {df['Dice'].std():.4f}")
print(f"  IoU    : {df['IoU'].mean():.4f} ± {df['IoU'].std():.4f}")

# ── Patient-level summary (each patient weighted equally) ─────────────────────
patient_df = df.groupby('patient')[['Dice', 'IoU']].mean()

print(f'\nPatient-level summary (n={len(patient_df)} patients):')
print(f"  Dice   : {patient_df['Dice'].mean():.4f} ± {patient_df['Dice'].std():.4f}")
print(f"  IoU    : {patient_df['IoU'].mean():.4f} ± {patient_df['IoU'].std():.4f}")

print('\nPer-patient Dice (sorted):')
print(patient_df['Dice'].sort_values().round(4).to_string())

# %%
# Build Case Map for visualisation
# Images: 2D_dataset_T1_png/{patient}/images/{stem}.png
# Masks : 2D_dataset_T1_png/{patient}/masks/{stem}.png
slice_to_img_path  = {}
slice_to_mask_path = {}
for patient_dir in PNG_DIR.iterdir():
    if not patient_dir.is_dir():
        continue
    pid = patient_dir.name
    for img_path in (patient_dir / 'images').glob('*.png'):
        slice_to_img_path[img_path.stem] = img_path
    for mask_path in (patient_dir / 'masks').glob('*.png'):
        slice_to_mask_path[mask_path.stem] = mask_path

case_map = {}
for fn in VAL_DIR.iterdir():
    if fn.suffix != '.png':
        continue
    stem = fn.stem
    img  = slice_to_img_path.get(stem)
    mask = slice_to_mask_path.get(stem)
    if not img or not mask:
        continue
    row = df[df['case'] == f'{stem}.png']
    case_map[stem] = {
        'patient': slice_to_patient.get(stem, 'unknown'),
        'pred':    fn,
        'image':   img,
        'mask':    mask,
        'dice':    row['Dice'].values[0] if len(row) else None,
        'iou':     row['IoU'].values[0]  if len(row) else None,
    }

print(f'\nTotal predictions : {len(list(VAL_DIR.glob("*.png")))}')
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
        axes[row][0].set_ylabel(f"pt={c['patient']}\n{dice_str}", fontsize=8)

        gt_overlay = np.zeros((*img.shape, 4), dtype=float)
        gt_overlay[gt_bin] = [1, 0, 0, 0.45]
        axes[row][1].imshow(img, cmap='gray')
        axes[row][1].imshow(gt_overlay)

        pred_overlay = np.zeros((*img.shape, 4), dtype=float)
        pred_overlay[pred_bin] = [0, 1, 0, 0.45]
        axes[row][2].imshow(img, cmap='gray')
        axes[row][2].imshow(pred_overlay)

        combined = np.zeros((*img.shape, 4), dtype=float)
        combined[gt_bin & ~pred_bin] = [1, 0, 0, 0.5]   # red    — FN
        combined[pred_bin & ~gt_bin] = [0, 1, 0, 0.5]   # green  — FP
        combined[gt_bin & pred_bin]  = [1, 1, 0, 0.6]   # yellow — TP
        axes[row][3].imshow(img, cmap='gray')
        axes[row][3].imshow(combined)

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.002)
    plt.tight_layout()
    plt.show()


# %%
def plot_patient(pid):
    patient_slices = sorted(
        [v for v in case_map.values() if v['patient'] == pid],
        key=lambda x: int(Path(x['pred']).stem)
    )
    if not patient_slices:
        return

    mean_dice = patient_df.loc[pid, 'Dice']
    mean_iou  = patient_df.loc[pid, 'IoU']
    n         = len(patient_slices)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ['MRI Image', 'Ground Truth (red)', 'Prediction (green)',
                  'GT vs Pred\n(red=FN  green=FP  yellow=TP)']
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, fontweight='bold')

    for row, c in enumerate(patient_slices):
        img      = np.array(Image.open(c['image']).convert('L'))
        gt_bin   = np.array(Image.open(c['mask']).convert('L')) > 0
        pred_bin = np.array(Image.open(c['pred']).convert('L')) > 0

        slice_id = Path(c['pred']).stem
        dice_str = f"Dice={c['dice']:.3f}  IoU={c['iou']:.3f}" if c['dice'] is not None else 'no metrics'
        axes[row][0].imshow(img, cmap='gray')
        axes[row][0].set_ylabel(f"slice {slice_id}\n{dice_str}", fontsize=8)

        gt_overlay = np.zeros((*img.shape, 4), dtype=float)
        gt_overlay[gt_bin] = [1, 0, 0, 0.45]
        axes[row][1].imshow(img, cmap='gray')
        axes[row][1].imshow(gt_overlay)

        pred_overlay = np.zeros((*img.shape, 4), dtype=float)
        pred_overlay[pred_bin] = [0, 1, 0, 0.45]
        axes[row][2].imshow(img, cmap='gray')
        axes[row][2].imshow(pred_overlay)

        combined = np.zeros((*img.shape, 4), dtype=float)
        combined[gt_bin & ~pred_bin] = [1, 0, 0, 0.5]   # red    — FN
        combined[pred_bin & ~gt_bin] = [0, 1, 0, 0.5]   # green  — FP
        combined[gt_bin & pred_bin]  = [1, 1, 0, 0.6]   # yellow — TP
        axes[row][3].imshow(img, cmap='gray')
        axes[row][3].imshow(combined)

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle(
        f'Patient {pid}  ({n} slices)  —  Mean Dice={mean_dice:.4f}  IoU={mean_iou:.4f}',
        fontsize=13, fontweight='bold', y=1.002
    )
    plt.tight_layout()
    plt.show()


# ── Worst 10 patients by mean Dice ───────────────────────────────────────────
print('=== WORST 10 PATIENTS ===')
for pid in patient_df['Dice'].sort_values().index[:10]:
    plot_patient(pid)

# ── Best 10 patients by mean Dice ────────────────────────────────────────────
print('=== BEST 10 PATIENTS ===')
for pid in patient_df['Dice'].sort_values(ascending=False).index[:10]:
    plot_patient(pid)

# %%
# ── Worst predictions (Dice below threshold) ──────────────────────────────────
THRESHOLD = 0.3

worst = sorted(
    [v for v in case_map.values() if v['dice'] is not None and v['dice'] < THRESHOLD],
    key=lambda x: x['dice']
)[:30]

print(f'Showing worst {len(worst)} cases (Dice < {THRESHOLD}):')
for i, c in enumerate(worst):
    print(f"  #{i+1:2d}  pt={c['patient']}  Dice={c['dice']:.4f}  IoU={c['iou']:.4f}  slice={Path(c['pred']).stem}")

if worst:
    _, axes = plt.subplots(len(worst), 4, figsize=(16, 4 * len(worst)))
    if len(worst) == 1:
        axes = [axes]
    col_titles = ['MRI Image', 'Ground Truth (red)', 'Prediction (green)',
                  'GT vs Pred\n(red=FN  green=FP  yellow=TP)']
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, fontweight='bold')

    for row, c in enumerate(worst):
        img      = np.array(Image.open(c['image']).convert('L'))
        gt_bin   = np.array(Image.open(c['mask']).convert('L')) > 0
        pred_bin = np.array(Image.open(c['pred']).convert('L')) > 0

        slice_id = Path(c['image']).stem
        dice_str = f"Dice={c['dice']:.3f}  IoU={c['iou']:.3f}"
        axes[row][0].imshow(img, cmap='gray')
        axes[row][0].set_ylabel(f"pt={c['patient']}\n{slice_id}  {dice_str}", fontsize=8)
        axes[row][0].text(4, 14, slice_id, color='yellow', fontsize=9, fontweight='bold')

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

    plt.suptitle(f'nnUNetTrainer_100epochs 2D fold_0 — Worst {len(worst)} Predictions (Dice < {THRESHOLD})',
                 fontsize=13, fontweight='bold', y=1.002)
    plt.tight_layout()
    plt.show()

# %%
