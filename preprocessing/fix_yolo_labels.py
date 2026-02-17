import os
from pathlib import Path
import csv

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\merged_dataset_v3"
LABEL_ROOT = os.path.join(DATA_ROOT, "labels")
OUT_REPORT = os.path.join(DATA_ROOT, "label_fix_report.csv")

SPLITS = ["train", "val", "test"]

# OLD → NEW mapping
# 0: glioma        → 0
# 1: meningioma    → 1
# 3: pituitary     → 2
# 2: no_tumor      → REMOVE
REMAP = {
    0: 0,
    1: 1,
    3: 2,
}

REMOVE_CLASS = 2  # no_tumor dummy boxes

# ============================================================
def process_label_file(path: str):
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    removed = 0
    remapped = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = int(parts[0])

        # Remove no_tumor dummy boxes
        if cls == REMOVE_CLASS:
            removed += 1
            continue

        # Remap pituitary
        if cls in REMAP:
            new_cls = REMAP[cls]
            if new_cls != cls:
                remapped += 1
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts) + "\n")

    # Write back
    with open(path, "w") as f:
        f.writelines(new_lines)

    return removed, remapped, len(new_lines)

# ============================================================
def main():
    rows = []

    for split in SPLITS:
        split_dir = os.path.join(LABEL_ROOT, split)
        if not os.path.exists(split_dir):
            continue

        for cls_folder in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls_folder)
            if not os.path.isdir(cls_dir):
                continue

            for file in os.listdir(cls_dir):
                if not file.endswith(".txt"):
                    continue

                path = os.path.join(cls_dir, file)

                removed, remapped, remaining = process_label_file(path)

                if removed > 0 or remapped > 0:
                    rows.append([
                        split,
                        cls_folder,
                        file,
                        removed,
                        remapped,
                        remaining
                    ])

    # Save report
    with open(OUT_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "folder_class",
            "label_file",
            "removed_no_tumor_boxes",
            "remapped_pituitary_boxes",
            "remaining_boxes"
        ])
        writer.writerows(rows)

    print("✅ LABEL FIX COMPLETE")
    print(f"📄 Report saved to: {OUT_REPORT}")

if __name__ == "__main__":
    main()
