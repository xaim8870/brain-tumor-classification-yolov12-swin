import os
from collections import defaultdict

DATASET_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Robofolo-Dataset"

def explore():
    summary = defaultdict(lambda: defaultdict(int))

    for split in ["train", "valid", "test"]:
        img_root = os.path.join(DATASET_ROOT, "images", split)
        lbl_root = os.path.join(DATASET_ROOT, "labels", split)

        print(f"\n📁 SPLIT: {split.upper()}")

        for cls in sorted(os.listdir(img_root)):
            img_cls = os.path.join(img_root, cls)
            lbl_cls = os.path.join(lbl_root, cls)

            imgs = [f for f in os.listdir(img_cls) if f.endswith((".jpg", ".png"))]
            lbls = [f for f in os.listdir(lbl_cls) if f.endswith(".txt")]

            summary[split][cls] = len(imgs)

            print(f"  {cls:<15} | Images: {len(imgs):<5} | Labels: {len(lbls):<5}")

    return summary

if __name__ == "__main__":
    explore()
