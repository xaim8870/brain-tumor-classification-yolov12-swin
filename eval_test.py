import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from torchvision import transforms

from datasets.brain_tumor_dataset import BrainTumorDataset
from models.yolov12_swin_classifier import YOLOv12SwinClassifier

# ===============================
# PATHS (EDIT IF NEEDED)
# ===============================
OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\figshare_class3"

BEST_CKPT = os.path.join(OUT_DIR, "checkpoints", "best.pt")
CLASS_NAMES_PATH = os.path.join(OUT_DIR, "class_names.json")

TEST_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Figshare-2\Figshare-2\dataset\test\images"

PLOTS_DIR = os.path.join(OUT_DIR, "post_eval_plots_test")
CSV_DIR   = os.path.join(OUT_DIR, "post_eval_csv_test")
PREDS_DIR = os.path.join(OUT_DIR, "preds_test")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# PLOT STYLE
# ===============================
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

def savefig(path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def style_legend():
    leg = plt.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(1.0)

# ===============================
# LOAD CLASS NAMES
# ===============================
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Missing: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)
print("Classes:", class_names)

# ===============================
# TRANSFORMS (SAFE DEFAULT)
# Your Swin is configured with img_size=512, so use 512.
# ===============================
val_tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===============================
# DATASET + LOADER
# IMPORTANT: your folder names are "Glioma", "Meningioma", "Pituitary"
# But in training you likely used lowercase names. So we normalise to lower.
# ===============================
test_ds = BrainTumorDataset(
    image_root=TEST_ROOT,
    transform=val_tf,
    class_name_fn=lambda s: s.lower()
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("TEST samples:", len(test_ds))
print("Dataset classes order:", test_ds.classes)
print("Dataset class_to_idx:", test_ds.class_to_idx)

# ===============================
# LOAD MODEL + CHECKPOINT
# ===============================
model = YOLOv12SwinClassifier(num_classes=num_classes)
ckpt = torch.load(BEST_CKPT, map_location="cpu")

# your training saves {"epoch":..., "model_state": ...}
model.load_state_dict(ckpt["model_state"], strict=True)

model.to(DEVICE)
model.eval()

# ===============================
# INFERENCE
# ===============================
all_true, all_pred, all_prob = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        all_true.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(probs.cpu().numpy())

y_true = np.concatenate(all_true).astype(int)
y_pred = np.concatenate(all_pred).astype(int)
y_prob = np.concatenate(all_prob)

wrong = int((y_true != y_pred).sum())
acc = float((y_true == y_pred).mean())

print("\n=== TEST RESULTS ===")
print("Wrong predictions:", wrong)
print("Accuracy:", acc)

# Save NPZ
npz_path = os.path.join(PREDS_DIR, "test_best.npz")
np.savez(npz_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
print("Saved NPZ:", npz_path)

# ===============================
# CONFUSION MATRIX (counts + normalized)
# ===============================
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

cm_csv = os.path.join(CSV_DIR, "cm_test_best_counts.csv")
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv)

plt.figure(figsize=(6.2, 5.2))
plt.imshow(cm, interpolation="nearest", cmap="Oranges")
plt.title("Confusion Matrix (Counts) | TEST (best.pt)")
plt.colorbar(fraction=0.046, pad=0.04)
ticks = np.arange(num_classes)
plt.xticks(ticks, class_names, rotation=25, ha="right")
plt.yticks(ticks, class_names)

thresh = cm.max() * 0.55 if cm.max() > 0 else 0.5
for i in range(num_classes):
    for j in range(num_classes):
        v = int(cm[i, j])
        plt.text(j, i, f"{v}", ha="center", va="center",
                 color="white" if v > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
cm_png = os.path.join(PLOTS_DIR, "cm_test_best_counts.png")
savefig(cm_png)

# Normalized
cm_norm = cm.astype(np.float64)
cm_norm = cm_norm / (cm_norm.sum(axis=1, keepdims=True) + 1e-12)

cmn_csv = os.path.join(CSV_DIR, "cm_test_best_normalized.csv")
pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(cmn_csv)

plt.figure(figsize=(6.2, 5.2))
plt.imshow(cm_norm, interpolation="nearest", cmap="Oranges")
plt.title("Confusion Matrix (Normalized) | TEST (best.pt)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(ticks, class_names, rotation=25, ha="right")
plt.yticks(ticks, class_names)

thresh = cm_norm.max() * 0.55 if cm_norm.max() > 0 else 0.5
for i in range(num_classes):
    for j in range(num_classes):
        v = cm_norm[i, j]
        plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                 color="white" if v > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
cmn_png = os.path.join(PLOTS_DIR, "cm_test_best_normalized.png")
savefig(cmn_png)

print("Saved CM:", cm_png, "and", cmn_png)

# ===============================
# ROC CURVES (OvR)
# ===============================
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

fpr, tpr, roc_auc = {}, {}, {}

for c in range(num_classes):
    fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
    roc_auc[c] = auc(fpr[c], tpr[c])

fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for c in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
mean_tpr /= num_classes

fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Save AUC CSV
auc_rows = [{"class": class_names[c], "auc": float(roc_auc[c])} for c in range(num_classes)]
auc_rows += [{"class": "micro_avg", "auc": float(roc_auc["micro"])},
             {"class": "macro_avg", "auc": float(roc_auc["macro"])}]

auc_csv = os.path.join(CSV_DIR, "roc_auc_test_best.csv")
pd.DataFrame(auc_rows).to_csv(auc_csv, index=False)

# Plot ROC
plt.figure(figsize=(6.6, 5.4))
plt.plot(fpr["micro"], tpr["micro"], linestyle="--",
         label=f"micro-average (AUC = {roc_auc['micro']:.3f})")
plt.plot(fpr["macro"], tpr["macro"], linestyle="--",
         label=f"macro-average (AUC = {roc_auc['macro']:.3f})")

for c in range(num_classes):
    plt.plot(fpr[c], tpr[c], label=f"{class_names[c]} (AUC = {roc_auc[c]:.3f})")

plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (OvR) | TEST (best.pt)")
style_legend()

roc_png = os.path.join(PLOTS_DIR, "roc_test_best.png")
savefig(roc_png)

print("Saved ROC:", roc_png)
print("Saved AUC CSV:", auc_csv)
