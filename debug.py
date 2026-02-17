import os, hashlib
from pathlib import Path

DATASET_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Figshare-2\Figshare-2\dataset"
IMG_EXTS = {".jpg",".jpeg",".png"}

def hash_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def collect(split):
    root = Path(DATASET_ROOT) / split / "images"
    hashes = {}
    for img in root.rglob("*"):
        if img.suffix.lower() in IMG_EXTS:
            hashes.setdefault(hash_file(img), []).append(str(img))
    return hashes

train_h = collect("train")
val_h   = collect("val")
test_h  = collect("test")

train_set = set(train_h.keys())
val_set   = set(val_h.keys())
test_set  = set(test_h.keys())

print("Train ∩ Val duplicates:", len(train_set & val_set))
print("Train ∩ Test duplicates:", len(train_set & test_set))
print("Val ∩ Test duplicates:", len(val_set & test_set))
