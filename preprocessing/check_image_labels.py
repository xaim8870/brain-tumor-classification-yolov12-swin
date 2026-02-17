import os

DATASET_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Robofolo-Dataset"

def check(split="train"):
    img_root = os.path.join(DATASET_ROOT, "images", split)
    lbl_root = os.path.join(DATASET_ROOT, "labels", split)

    missing = []

    for cls in os.listdir(img_root):
        img_cls = os.path.join(img_root, cls)
        lbl_cls = os.path.join(lbl_root, cls)

        img_files = {os.path.splitext(f)[0] for f in os.listdir(img_cls)}
        lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_cls)}

        diff = img_files - lbl_files
        if diff:
            missing.extend([f"{cls}/{d}" for d in diff])

    print(f"\n❌ Missing labels: {len(missing)}")
    for m in missing[:5]:
        print(" ", m)

if __name__ == "__main__":
    check("train")
