# data_tools/rebuild_splits_v4.py
import os
import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict

from PIL import Image
import imagehash

# =========================
# CONFIG
# =========================
IN_ROOT  = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\merged_dataset_v3"
OUT_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\merged_dataset_v4"

SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# pHash distance threshold for "near duplicate"
# 0 = identical, 5-10 often similar. Start conservative.
PHASH_MAX_DIST = 6

# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_images(root):
    root = Path(root)
    for split_dir in root.glob("images/*/*"):
        # split_dir = images/{split}/{class}
        if not split_dir.is_dir():
            continue
        for p in split_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                yield p

def get_class_from_path(p: Path):
    # ...\images\train\glioma\file.jpg -> glioma
    return p.parent.name

def compute_phash(img_path: Path):
    # robust to resize/re-encode
    img = Image.open(img_path).convert("RGB")
    return imagehash.phash(img)

def copy_img(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))

# =========================
def main():
    random.seed(SEED)

    # 1) Collect all images by class from ALL splits in v3
    by_class = defaultdict(list)
    for p in iter_images(IN_ROOT):
        by_class[get_class_from_path(p)].append(p)

    print("Classes found:", list(by_class.keys()))
    print("Counts:", {k: len(v) for k, v in by_class.items()})

    # 2) Exact dedup by sha256
    seen_sha = {}
    unique_by_class = defaultdict(list)
    exact_dups = 0

    for cls, files in by_class.items():
        for p in files:
            h = sha256_file(p)
            if h in seen_sha:
                exact_dups += 1
                continue
            seen_sha[h] = str(p)
            unique_by_class[cls].append(p)

    print(f"✅ Exact duplicates removed: {exact_dups}")

    # 3) Near-dup clustering by pHash
    # We prevent near-duplicates from going into different splits.
    # Strategy: build clusters per class using phash distance.
    clusters_by_class = defaultdict(list)

    for cls, files in unique_by_class.items():
        phashes = []
        for p in files:
            try:
                phashes.append((p, compute_phash(p)))
            except Exception:
                # unreadable image -> skip
                continue

        clusters = []  # list of list[Path]
        for p, h in phashes:
            placed = False
            for c in clusters:
                # compare with cluster representative only (fast)
                rep = c[0]["hash"]
                if (h - rep) <= PHASH_MAX_DIST:
                    c.append({"path": p, "hash": h})
                    placed = True
                    break
            if not placed:
                clusters.append([{"path": p, "hash": h}])

        # store as list of clusters (paths only)
        clusters_by_class[cls] = [[x["path"] for x in c] for c in clusters]
        print(f"{cls}: {len(files)} images -> {len(clusters)} near-dup clusters")

    # 4) Split by clusters (NOT by images) to avoid leakage
    out_img_root = Path(OUT_ROOT) / "images"
    for split in ["train", "val", "test"]:
        for cls in clusters_by_class.keys():
            ensure_dir(out_img_root / split / cls)

    for cls, clusters in clusters_by_class.items():
        random.shuffle(clusters)

        n = len(clusters)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])
        # remainder to test
        train_c = clusters[:n_train]
        val_c = clusters[n_train:n_train + n_val]
        test_c = clusters[n_train + n_val:]

        assign = [("train", train_c), ("val", val_c), ("test", test_c)]

        for split, split_clusters in assign:
            for cluster in split_clusters:
                for src in cluster:
                    dst = out_img_root / split / cls / src.name
                    copy_img(src, dst)

        print(f"[{cls}] clusters: train={len(train_c)} val={len(val_c)} test={len(test_c)}")

    print("\n✅ merged_dataset_v4 created at:", OUT_ROOT)
    print("Now train on v4.")

if __name__ == "__main__":
    main()
