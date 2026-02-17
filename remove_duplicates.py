import os, hashlib, shutil
from pathlib import Path

DATASET_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Figshare-2\Figshare-2\dataset"
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_images(split):
    root = Path(DATASET_ROOT) / split / "images"
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            yield p

def label_path(img_path: Path):
    # train/images/Class/x.jpg -> train/labels/Class/x.txt
    return Path(str(img_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")

# 1) Hash all TRAIN images
train_hashes = {}
for p in iter_images("train"):
    train_hashes[md5(p)] = p

print("Train hashes:", len(train_hashes))

# 2) Remove duplicates from VAL + TEST if already in train
removed = {"val": 0, "test": 0}
for split in ["val", "test"]:
    for p in list(iter_images(split)):
        h = md5(p)
        if h in train_hashes:
            # delete val/test image + its label
            lbl = label_path(p)
            try:
                p.unlink()
            except: pass
            if lbl.exists():
                try: lbl.unlink()
                except: pass
            removed[split] += 1

print("Removed duplicates found in train -> val:", removed["val"])
print("Removed duplicates found in train -> test:", removed["test"])

# 3) Remove duplicates between VAL and TEST (after step 2)
val_hashes = {}
for p in iter_images("val"):
    val_hashes[md5(p)] = p

removed_vt = 0
for p in list(iter_images("test")):
    h = md5(p)
    if h in val_hashes:
        lbl = label_path(p)
        try:
            p.unlink()
        except: pass
        if lbl.exists():
            try: lbl.unlink()
            except: pass
        removed_vt += 1

print("Removed duplicates val<->test:", removed_vt)
print("✅ Done. Re-run EDA + training now.")
