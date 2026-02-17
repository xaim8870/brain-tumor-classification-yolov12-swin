import os
from collections import Counter

ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Figshare-2\Figshare-2\dataset"
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    img_root = os.path.join(ROOT, split, "images")
    c = Counter()
    total = 0
    for cls in os.listdir(img_root):
        cls_dir = os.path.join(img_root, cls)
        if not os.path.isdir(cls_dir): 
            continue
        n = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg",".png",".jpeg"))])
        c[cls] += n
        total += n
    print(split, total, dict(c))
