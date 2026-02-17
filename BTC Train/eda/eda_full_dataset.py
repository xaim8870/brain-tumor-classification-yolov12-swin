import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Figshare-3\Figshare-3\dataset"
OUT_DIR = os.path.join(DATA_ROOT, "eda_report")
SPLITS = ["train", "val", "test"]

TARGET_CLASSES = ["glioma", "meningioma", "pituitary"]  # logical names (lowercase)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# AUTO-DETECT STRUCTURE
# ============================================================
# Case A (Roboflow): DATA_ROOT/images/train/...
roboflow_images_root = os.path.join(DATA_ROOT, "images")
roboflow_labels_root = os.path.join(DATA_ROOT, "labels")

# Case B (Figshare): DATA_ROOT/train/images/...
figshare_probe = os.path.join(DATA_ROOT, "train", "images")

if os.path.exists(os.path.join(roboflow_images_root, "train")):
    STRUCT = "roboflow"
    def split_img_root(split): return os.path.join(DATA_ROOT, "images", split)
    def split_lbl_root(split): return os.path.join(DATA_ROOT, "labels", split)
elif os.path.exists(figshare_probe):
    STRUCT = "figshare"
    def split_img_root(split): return os.path.join(DATA_ROOT, split, "images")
    def split_lbl_root(split): return os.path.join(DATA_ROOT, split, "labels")
else:
    raise FileNotFoundError(
        "Could not detect dataset structure. Expected either:\n"
        "A) DATA_ROOT/images/train/...\n"
        "B) DATA_ROOT/train/images/...\n"
        f"DATA_ROOT={DATA_ROOT}"
    )

print(f"✅ Detected structure: {STRUCT}")

# ============================================================
# HELPERS
# ============================================================
def list_images(folder):
    if not os.path.exists(folder):
        return []
    out = []
    for r, _, fs in os.walk(folder):
        for f in fs:
            if Path(f).suffix.lower() in IMG_EXTS:
                out.append(os.path.join(r, f))
    return out

def read_yolo_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # ---- Case 1: standard YOLO box (5 numbers) ----
            # expected: class xc yc w h
            if len(parts) == 5:
                c, x, y, w, h = parts
                try:
                    cid = int(c)
                except ValueError:
                    # handles "0.0"
                    cid = int(float(c))

                boxes.append((cid, float(x), float(y), float(w), float(h)))
                continue

            # ---- Case 2: YOLO segmentation (class + polygon coords) ----
            # format: class x1 y1 x2 y2 ...
            # We can skip these for bbox stats OR approximate bbox from polygon.
            # For now: skip but log
            # (Uncomment if you want bbox approximation)
            # if len(parts) > 5 and len(parts) % 2 == 1:
            #     try:
            #         cid = int(float(parts[0]))
            #         coords = list(map(float, parts[1:]))
            #         xs = coords[0::2]
            #         ys = coords[1::2]
            #         x_min, x_max = min(xs), max(xs)
            #         y_min, y_max = min(ys), max(ys)
            #         xc = (x_min + x_max) / 2
            #         yc = (y_min + y_max) / 2
            #         bw = (x_max - x_min)
            #         bh = (y_max - y_min)
            #         boxes.append((cid, xc, yc, bw, bh))
            #     except Exception:
            #         pass
            #     continue

            # Otherwise ignore unknown lines (or log)
    return boxes

def draw_boxes(img, boxes):
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def safe_listdir(p):
    return os.listdir(p) if os.path.exists(p) else []

# ============================================================
# EDA STORAGE
# ============================================================
rows_integrity = []
rows_bbox = []
rows_sizes = []
rows_counts = []
ignored_classes_found = set()

# ============================================================
# MAIN SCAN
# ============================================================
for split in SPLITS:
    img_root = split_img_root(split)
    lbl_root = split_lbl_root(split)

    if not os.path.exists(img_root):
        rows_integrity.append([split, "", img_root, "missing_images_root"])
        continue

    found_class_folders = safe_listdir(img_root)

    # map found folders to lowercase for matching
    folder_map = {c.lower(): c for c in found_class_folders}

    for c in found_class_folders:
        if c.lower() not in set(TARGET_CLASSES):
            ignored_classes_found.add(c)

    for cls_lower in TARGET_CLASSES:
        if cls_lower not in folder_map:
            rows_integrity.append([split, cls_lower, img_root, "missing_class_folder"])
            continue

        cls_folder = folder_map[cls_lower]
        img_dir = os.path.join(img_root, cls_folder)
        lbl_dir = os.path.join(lbl_root, cls_folder)

        images = list_images(img_dir)
        rows_counts.append([split, cls_folder, len(images)])

        for img_path in images:
            stem = Path(img_path).stem
            lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

            img = cv2.imread(img_path)
            if img is None:
                rows_integrity.append([split, cls_folder, img_path, "corrupt_image"])
                continue

            H, W = img.shape[:2]
            rows_sizes.append([split, cls_folder, W, H, (W / H) if H else np.nan])

            boxes = read_yolo_label(lbl_path)

            if not os.path.exists(lbl_path):
                rows_integrity.append([split, cls_folder, img_path, "missing_label_file"])

            for (cid, xc, yc, bw, bh) in boxes:
                rows_bbox.append([
                    split, cls_folder, img_path,
                    cid, bw, bh, bw * bh
                ])

# ============================================================
# SAVE CSVs
# ============================================================
df_counts = pd.DataFrame(rows_counts, columns=["split", "class", "num_images"])
df_integrity = pd.DataFrame(rows_integrity, columns=["split", "folder_class", "path", "issue"])
df_bbox = pd.DataFrame(rows_bbox, columns=["split", "folder_class", "image", "class_id", "bbox_w", "bbox_h", "bbox_area"])
df_sizes = pd.DataFrame(rows_sizes, columns=["split", "folder_class", "width", "height", "aspect_ratio"])

df_counts.to_csv(os.path.join(OUT_DIR, "class_counts.csv"), index=False)
df_integrity.to_csv(os.path.join(OUT_DIR, "integrity_issues.csv"), index=False)
df_bbox.to_csv(os.path.join(OUT_DIR, "bbox_stats.csv"), index=False)
df_sizes.to_csv(os.path.join(OUT_DIR, "image_size_stats.csv"), index=False)

# ============================================================
# PLOTS (guard for empties)
# ============================================================
if not df_bbox.empty and (df_bbox["bbox_area"] > 0).any():
    plt.figure(figsize=(8, 5))
    df_bbox[df_bbox["bbox_area"] > 0]["bbox_area"].hist(bins=50)
    plt.title("Bounding Box Area Distribution")
    plt.xlabel("Normalized Area")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bbox_area_distribution.png"), dpi=300)
    plt.close()
else:
    print("⚠️ No bbox data found. Check label paths / structure.")

if not df_counts.empty:
    pivot = df_counts.pivot(index="class", columns="split", values="num_images").fillna(0)
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.title("Images per Class per Split")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_counts_bar.png"), dpi=300)
    plt.close()

if not df_sizes.empty:
    plt.figure(figsize=(7, 5))
    plt.scatter(df_sizes["width"], df_sizes["height"], alpha=0.3)
    plt.title("Image Width vs Height")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "image_size_scatter.png"), dpi=300)
    plt.close()

# ============================================================
# VISUAL SAMPLES
# ============================================================
def make_samples_grid(cls_lower, k=16):
    imgs = []
    for split in SPLITS:
        img_root = split_img_root(split)
        found = safe_listdir(img_root)
        folder_map = {c.lower(): c for c in found}
        if cls_lower not in folder_map:
            continue
        cls_folder = folder_map[cls_lower]
        imgs.extend(list_images(os.path.join(img_root, cls_folder)))

    random.shuffle(imgs)
    imgs = imgs[:k]
    if not imgs:
        return

    grid = []
    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        grid.append(img)

    while len(grid) % 4 != 0:
        grid.append(np.zeros((256, 256, 3), dtype=np.uint8))

    rows = [np.hstack(grid[i:i+4]) for i in range(0, len(grid), 4)]
    grid_img = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT_DIR, f"samples_{cls_lower}.png"), grid_img)

for cls in TARGET_CLASSES:
    make_samples_grid(cls, k=16)

print("✅ EDA COMPLETE")
print(f"📁 Reports saved to: {OUT_DIR}")
if ignored_classes_found:
    print(f"⚠️ Ignored class folders found: {sorted(list(ignored_classes_found))}")
