# %% [markdown]
# # nnUNet Feature Map Visualisation — fold_0
# Shows mean activation at each encoder and decoder stage for selected slices.

# %%
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

# %%
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path('/Users/namnguyen/PycharmProjects/brain_seg_research')
CKPT        = BASE_DIR / 'nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_0/checkpoint_final.pth'
PNG_DIR     = BASE_DIR / 'datasets/2D_dataset_T1_png'
VAL_DIR     = BASE_DIR / 'nnUNet_results/Dataset001_BrainTumor/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_0/validation'
OUT_DIR     = BASE_DIR / 'evaluation/feature_maps_output'
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# ── Load model from checkpoint plans (no hardcoded params) ───────────────────
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)

plans = ckpt['init_args']['plans']
arch  = plans['configurations']['2d']['architecture']

model = get_network_from_plans(
    arch_class_name=arch['network_class_name'],
    arch_kwargs=arch['arch_kwargs'],
    arch_kwargs_req_import=arch['_kw_requires_import'],
    input_channels=1,
    output_channels=2,
    deep_supervision=True,
)
model.load_state_dict(ckpt['network_weights'])
model.to(DEVICE)
model.eval()
print('Model loaded:', type(model).__name__)

# %%
# ── Build slice → path maps ───────────────────────────────────────────────────
slice_to_img     = {}
slice_to_mask    = {}
slice_to_patient = {}
for patient_dir in PNG_DIR.iterdir():
    if not patient_dir.is_dir():
        continue
    pid = patient_dir.name
    for p in (patient_dir / 'images').glob('*.png'):
        slice_to_img[p.stem]     = p
        slice_to_patient[p.stem] = pid
    for p in (patient_dir / 'masks').glob('*.png'):
        slice_to_mask[p.stem] = p

# %%
# ── Preprocessing (mirrors nnUNet ZScoreNormalization) ────────────────────────
MEAN = 87.8178
STD  = 36.7825

def preprocess(img_path):
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
    img = (img - MEAN) / STD
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)

# %%
# ── Register forward hooks on all encoder + decoder stages ────────────────────
feature_maps = {}

def make_hook(name):
    def hook(module, inp, out):
        # out may be a tensor or tuple; take first element if needed
        act = out[0] if isinstance(out, (list, tuple)) else out
        feature_maps[name] = act.detach().cpu()
    return hook

hooks = []
for i, stage in enumerate(model.encoder.stages):
    hooks.append(stage.register_forward_hook(make_hook(f'enc_{i}')))
for i, stage in enumerate(model.decoder.stages):
    hooks.append(stage.register_forward_hook(make_hook(f'dec_{i}')))

print(f'Registered {len(hooks)} hooks  '
      f'({len(model.encoder.stages)} encoder + {len(model.decoder.stages)} decoder stages)')

# %%
# ── Visualise feature maps for a given slice ──────────────────────────────────
def visualise_feature_maps(stem, title_prefix=''):
    img_path  = slice_to_img.get(stem)
    mask_path = slice_to_mask.get(stem)
    pred_path = VAL_DIR / f'{stem}.png'

    if not img_path or not mask_path or not pred_path.exists():
        print(f'Missing files for slice {stem}')
        return

    # Forward pass
    feature_maps.clear()
    x = preprocess(img_path)
    with torch.no_grad():
        _ = model(x)

    # Raw arrays for display
    img_np   = np.array(Image.open(img_path).convert('L'))
    mask_np  = np.array(Image.open(mask_path).convert('L')) > 0
    pred_np  = np.array(Image.open(pred_path).convert('L')) > 0

    enc_feats = plans['configurations']['2d']['architecture']['arch_kwargs']['features_per_stage']
    dec_feats = list(reversed(enc_feats[:-1]))
    stage_names = (
        [f'Enc {i}\n({enc_feats[i]}ch)' for i in range(len(enc_feats))] +
        [f'Dec {i}\n({dec_feats[i]}ch)' for i in range(len(dec_feats))]
    )
    fmap_keys = [f'enc_{i}' for i in range(8)] + [f'dec_{i}' for i in range(7)]

    n_stages = len(fmap_keys)
    fig, axes = plt.subplots(3, n_stages + 1, figsize=(3 * (n_stages + 1), 9))

    # ── Row 0: MRI / GT / Pred reference ─────────────────────────────────────
    for ax in axes[0]:
        ax.axis('off')
    axes[0][0].imshow(img_np, cmap='gray')
    axes[0][0].set_title('MRI', fontsize=8)

    gt_ov = np.zeros((*img_np.shape, 4), dtype=float)
    gt_ov[mask_np] = [1, 0, 0, 0.5]
    axes[0][1].imshow(img_np, cmap='gray')
    axes[0][1].imshow(gt_ov)
    axes[0][1].set_title('GT (red)', fontsize=8)

    pr_ov = np.zeros((*img_np.shape, 4), dtype=float)
    pr_ov[pred_np] = [0, 1, 0, 0.5]
    axes[0][2].imshow(img_np, cmap='gray')
    axes[0][2].imshow(pr_ov)
    axes[0][2].set_title('Pred (green)', fontsize=8)

    # ── Row 1: mean activation heatmaps ──────────────────────────────────────
    for col, (key, name) in enumerate(zip(fmap_keys, stage_names)):
        fmap = feature_maps[key][0]          # (C, H, W)
        mean_act = fmap.mean(dim=0).numpy()  # (H, W)
        axes[1][col + 1].imshow(mean_act, cmap='hot', interpolation='nearest')
        axes[1][col + 1].set_title(name, fontsize=7)
        axes[1][col + 1].axis('off')
    axes[1][0].imshow(img_np, cmap='gray')
    axes[1][0].set_title('MRI', fontsize=8)
    axes[1][0].axis('off')

    # ── Row 2: activation overlaid on MRI (normalised per stage) ─────────────
    for col, (key, name) in enumerate(zip(fmap_keys, stage_names)):
        fmap = feature_maps[key][0]
        mean_act = fmap.mean(dim=0).numpy()
        # resize to MRI resolution for overlay
        from PIL import Image as PILImage
        act_img = PILImage.fromarray(
            ((mean_act - mean_act.min()) / (mean_act.max() - mean_act.min() + 1e-8) * 255).astype(np.uint8)
        ).resize((img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR)
        act_arr = np.array(act_img) / 255.0

        axes[2][col + 1].imshow(img_np, cmap='gray')
        axes[2][col + 1].imshow(act_arr, cmap='hot', alpha=0.5, interpolation='nearest')
        axes[2][col + 1].axis('off')
    axes[2][0].imshow(img_np, cmap='gray')
    axes[2][0].axis('off')

    # Row labels
    for row_label, ax in zip(['Reference', 'Mean activation', 'Overlay on MRI'],
                              [axes[0][0], axes[1][0], axes[2][0]]):
        ax.set_ylabel(row_label, fontsize=9, fontweight='bold', rotation=0,
                      labelpad=60, va='center')

    plt.suptitle(f'{title_prefix}Slice {stem}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = OUT_DIR / f'feature_maps_{stem}.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# %%
# ── Pick representative slices from validation set ────────────────────────────
import json, pandas as pd

SUMMARY = VAL_DIR / 'summary.json'
with open(SUMMARY) as f:
    s = json.load(f)

records = []
for case in s['metric_per_case']:
    stem = Path(case['reference_file']).stem
    records.append({'stem': stem, 'Dice': case['metrics']['1']['Dice']})
df_val = pd.DataFrame(records).sort_values('Dice')

# Filter to slices that actually have tumor in GT
def has_tumor(stem):
    p = slice_to_mask.get(stem)
    return p is not None and np.array(Image.open(p).convert('L')).max() > 0

df_val = df_val[df_val['stem'].apply(has_tumor)].reset_index(drop=True)

# Add patient column then keep only the worst slice per patient
df_val['patient'] = df_val['stem'].map(slice_to_patient)
worst_per_patient = (
    df_val.sort_values('Dice')
          .drop_duplicates(subset='patient', keep='first')
          .head(30)
          .reset_index(drop=True)
)
print(f'Generating feature maps for {len(worst_per_patient)} worst unique patients...')

# %%
for rank, (_, row) in enumerate(worst_per_patient.iterrows(), start=1):
    print(f'[{rank}/{len(worst_per_patient)}] patient {row["patient"]}  '
          f'slice {row["stem"]}  Dice={row["Dice"]:.4f}')
    visualise_feature_maps(
        row['stem'],
        title_prefix=f'[WORST {rank}  pt={row["patient"]}]  '
    )

# %%
# ── Cleanup hooks ─────────────────────────────────────────────────────────────
for h in hooks:
    h.remove()
