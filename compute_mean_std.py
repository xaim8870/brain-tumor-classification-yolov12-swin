import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.mri_preprocess import brain_margin_crop

# ============================================================
# CONFIG (Figshare 3-class dataset)
# ============================================================
DATA_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Figshare-2\Figshare-2\dataset"

# Figshare structure: dataset/train/images/<ClassName>/*.jpg
IMG_ROOT = os.path.join(DATA_ROOT, "train", "images")

IMG_SIZE = 512
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ============================================================
def list_images(folder):
    imgs = []
    if not os.path.exists(folder):
        return imgs

    for r, _, fs in os.walk(folder):
        for f in fs:
            if Path(f).suffix.lower() in IMG_EXTS:
                imgs.append(os.path.join(r, f))
    return imgs

# ============================================================
def main():
    image_paths = list_images(IMG_ROOT)
    assert len(image_paths) > 0, f"❌ No training images found at: {IMG_ROOT}"

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    print(f"🔍 Computing mean/std on {len(image_paths)} images...")
    print(f"📁 Train root: {IMG_ROOT}")

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1) Margin crop (same as your training)
        img, _ = brain_margin_crop(img)

        # Safety (in case crop fails)
        if img is None or img.size == 0:
            continue

        # 2) Resize to 512×512
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # 3) Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Accumulate
        channel_sum += img.sum(axis=(0, 1))
        channel_sum_sq += (img ** 2).sum(axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sum_sq / pixel_count - mean ** 2)

    print("\n✅ FINAL DATASET NORMALIZATION (USE THESE IN augmentations.py):")
    print(f"MEAN = {mean.tolist()}")
    print(f"STD  = {std.tolist()}")

if __name__ == "__main__":
    main()
