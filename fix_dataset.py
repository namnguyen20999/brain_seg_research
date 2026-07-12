import os
import glob
import numpy as np
from PIL import Image

# 1. Define your paths
base_dir = "/workspace/brain_seg_research/nnUNet_raw/Dataset001_BrainTumorProcessed"
images_dir = os.path.join(base_dir, "imagesTr")
labels_dir = os.path.join(base_dir, "labelsTr")

print("Starting dataset fix...")

# 2. Fix Image Naming (Add _0000 if missing)
image_files = glob.glob(os.path.join(images_dir, "*.png"))
for img_path in image_files:
    filename = os.path.basename(img_path)
    if not filename.endswith("_0000.png"):
        new_name = filename.replace(".png", "_0000.png")
        os.rename(img_path, os.path.join(images_dir, new_name))
        
print("Images checked and renamed to end with _0000.png")

# 3. Fix Label Naming (Remove _0000 if accidentally added) and Fix Pixels
label_files = glob.glob(os.path.join(labels_dir, "*.png"))
for lbl_path in label_files:
    # Rename if needed
    filename = os.path.basename(lbl_path)
    if filename.endswith("_0000.png"):
        new_name = filename.replace("_0000.png", ".png")
        new_lbl_path = os.path.join(labels_dir, new_name)
        os.rename(lbl_path, new_lbl_path)
        lbl_path = new_lbl_path # Update path for pixel fixing
        
    # Open image, fix pixels, and save
    img = Image.open(lbl_path)
    img_array = np.array(img)
    
    # If masks use 255 for tumor, convert it to 1
    if np.max(img_array) > 1:
        img_array[img_array > 0] = 1
        fixed_img = Image.fromarray(img_array.astype(np.uint8))
        fixed_img.save(lbl_path)

print("Labels checked, renamed to match without _0000, and pixel values normalized to 0 and 1.")
print("Dataset is ready for nnU-Net!")
