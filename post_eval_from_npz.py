import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ===============================
# PATHS (EDIT THIS ONLY)
# ===============================
OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\figshare_class3"

METRICS_CSV = os.path.join(OUT_DIR, "metrics_global.csv")
PREDS_DIR   = os.path.join(OUT_DIR, "preds")
CLASS_NAMES_PATH = os.path.join(OUT_DIR, "class_names.json")

PLOTS_DIR = os.path.join(OUT_DIR, "post_eval_plots")
CSV_DIR   = os.path.join(OUT_DIR, "post_eval_csv")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ===============================
# PAPER STYLE
# ===============================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

def style_legend():
    leg = plt.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(1.0)

def savefig(path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)

# ===============================
# LOAD CLASS NAMES
# ===============================
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
else:
    raise FileNotFoundError(f"Missing class_names.json at: {CLASS_NAMES_PATH}")

num_classes = len(class_names)

# ===============================
# PICK BEST EPOCH FROM METRICS CSV
# ===============================
if not os.path.exists(METRICS_CSV):
    raise FileNotFoundError(f"Missing metrics_global.csv at: {METRICS_CSV}")

df = pd.read_csv(METRICS_CSV)
if "val_f1_macro" not in df.columns:
    raise ValueError("metrics_global.csv does not contain 'val_f1_macro'")

best_epoch = int(df.loc[df["val_f1_macro"].idxmax(), "epoch"])
npz_path = os.path.join(PREDS_DIR, f"val_epoch_{best_epoch:03d}.npz")

if not os.path.exists(npz_path):
    raise FileNotFoundError(
        f"Could not find best-epoch npz: {npz_path}\n"
        f"Check your preds folder naming (val_epoch_XXX.npz)."
    )

# ===============================
# LOAD BEST-EPOCH PREDICTIONS
# ===============================
data = np.load(npz_path)

if "y_true" not in data or "y_pred" not in data:
    raise KeyError("NPZ must contain at least: y_true and y_pred.")

y_true = data["y_true"].astype(int)
y_pred = data["y_pred"].astype(int)

# Print keys (helps debug ROC availability)
print("NPZ keys found:", list(data.keys()))

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

cm_csv = os.path.join(CSV_DIR, f"cm_best_epoch_{best_epoch:03d}.csv")
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv)

# --- Counts plot ---
plt.figure(figsize=(6.2, 5.2))
plt.imshow(cm, interpolation="nearest", cmap="Oranges")
plt.title(f"Confusion Matrix (Counts) | Best Epoch {best_epoch:03d}")
plt.colorbar(fraction=0.046, pad=0.04)

ticks = np.arange(num_classes)
plt.xticks(ticks, class_names, rotation=25, ha="right")
plt.yticks(ticks, class_names)

thresh = cm.max() * 0.55 if cm.max() > 0 else 0.5
for i in range(num_classes):
    for j in range(num_classes):
        val = int(cm[i, j])
        plt.text(j, i, f"{val}",
                 ha="center", va="center",
                 color="white" if val > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
counts_png = os.path.join(PLOTS_DIR, f"cm_best_epoch_{best_epoch:03d}_counts.png")
savefig(counts_png)

# --- Normalized plot ---
cm_norm = cm.astype(np.float64)
cm_norm = cm_norm / (cm_norm.sum(axis=1, keepdims=True) + 1e-12)

plt.figure(figsize=(6.2, 5.2))
plt.imshow(cm_norm, interpolation="nearest", cmap="Oranges")
plt.title(f"Confusion Matrix (Normalized) | Best Epoch {best_epoch:03d}")
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks(ticks, class_names, rotation=25, ha="right")
plt.yticks(ticks, class_names)

thresh = cm_norm.max() * 0.55 if cm_norm.max() > 0 else 0.5
for i in range(num_classes):
    for j in range(num_classes):
        val = cm_norm[i, j]
        plt.text(j, i, f"{val:.2f}",
                 ha="center", va="center",
                 color="white" if val > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
norm_png = os.path.join(PLOTS_DIR, f"cm_best_epoch_{best_epoch:03d}_normalized.png")
savefig(norm_png)

# ===============================
# ROC CURVES (needs scores/probabilities)
# ===============================
# We accept any of these keys for per-class scores:
# - y_prob   (already probabilities)
# - y_logits (we will softmax)
# - y_score  (assumed probabilities or scores; if looks unnormalised, we softmax)
score_key = None
for k in ["y_prob", "y_logits", "y_score", "y_scores", "probs", "logits"]:
    if k in data:
        score_key = k
        break

if score_key is None:
    print("\n⚠️ ROC skipped: no probability/score array found in NPZ.")
    print("Needed one of: y_prob / y_logits / y_score (shape [N, C]).")
    print("Confusion matrix outputs are saved successfully.")
else:
    scores = np.array(data[score_key])

    if scores.ndim != 2 or scores.shape[1] != num_classes:
        raise ValueError(
            f"Score array '{score_key}' must have shape (N, {num_classes}), "
            f"but got {scores.shape}."
        )

    # Convert to probabilities if needed
    if score_key in ["y_logits", "logits"]:
        y_prob = softmax(scores, axis=1)
    else:
        # If it doesn't look like probabilities, softmax it
        row_sums = scores.sum(axis=1)
        looks_like_prob = np.all(scores >= 0) and np.all(scores <= 1.0 + 1e-6) and np.allclose(row_sums, 1.0, atol=1e-3)
        y_prob = scores if looks_like_prob else softmax(scores, axis=1)

    # Binarize labels for OvR ROC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))  # (N, C)

    fpr = {}
    tpr = {}
    roc_auc = {}

    # Per-class ROC
    for c in range(num_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average ROC (average TPR over a common FPR grid)
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Save AUC table
    auc_rows = []
    for c in range(num_classes):
        auc_rows.append({"class": class_names[c], "auc": float(roc_auc[c])})
    auc_rows.append({"class": "micro_avg", "auc": float(roc_auc["micro"])})
    auc_rows.append({"class": "macro_avg", "auc": float(roc_auc["macro"])})

    auc_csv = os.path.join(CSV_DIR, f"roc_auc_best_epoch_{best_epoch:03d}.csv")
    pd.DataFrame(auc_rows).to_csv(auc_csv, index=False)

    # Plot ROC (per-class + micro + macro)
    plt.figure(figsize=(6.6, 5.4))
    plt.plot(fpr["micro"], tpr["micro"], linestyle="--",
             label=f"micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], linestyle="--",
             label=f"macro-average (AUC = {roc_auc['macro']:.3f})")

    for c in range(num_classes):
        plt.plot(fpr[c], tpr[c],
                 label=f"{class_names[c]} (AUC = {roc_auc[c]:.3f})")

    plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (OvR) | Best Epoch {best_epoch:03d}")
    style_legend()

    roc_png = os.path.join(PLOTS_DIR, f"roc_best_epoch_{best_epoch:03d}.png")
    savefig(roc_png)

    print("\n✅ Confusion matrix + ROC generated from preds (NO retraining).")
    print(f"Best epoch: {best_epoch}")
    print(f"Loaded: {npz_path}")
    print("Saved CM CSV:", cm_csv)
    print("Saved CM plots:", counts_png, "and", norm_png)
    print("Saved ROC plot:", roc_png)
    print("Saved ROC AUC CSV:", auc_csv)
