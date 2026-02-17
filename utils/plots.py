import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# =========================================================
# PAPER-QUALITY MATPLOTLIB STYLE
# =========================================================
def set_paper_style():
    """
    Paper-quality plot defaults:
    - consistent fonts
    - clean grid
    - high DPI handled in savefig
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _finalize_and_save(out_path, dpi=300):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# =========================================================
# TRAIN/VAL CURVES
# =========================================================
def plot_accuracy(global_csv_df, out_path):
    set_paper_style()
    plt.figure(figsize=(6.4, 4.0))
    # Train solid, Val dashed (paper standard)
    plt.plot(global_csv_df["epoch"], global_csv_df["train_accuracy"],
             color="blue", linewidth=2.0, label="Train")
    plt.plot(global_csv_df["epoch"], global_csv_df["val_accuracy"],
             color="orange", linewidth=2.0, linestyle="--", label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.grid(True, alpha=0.20)
    plt.legend(frameon=False)
    _finalize_and_save(out_path, dpi=300)


def plot_loss(global_csv_df, out_path):
    set_paper_style()
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(global_csv_df["epoch"], global_csv_df["train_loss"],
             color="blue", linewidth=2.0, label="Train")
    plt.plot(global_csv_df["epoch"], global_csv_df["val_loss"],
             color="orange", linewidth=2.0, linestyle="--", label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.20)
    plt.legend(frameon=False)
    _finalize_and_save(out_path, dpi=300)


# =========================================================
# CONFUSION MATRIX (counts + optional normalization)
# =========================================================
def plot_confusion_matrix(cm, class_names, out_path, normalize=False):
    """
    normalize=False -> raw counts
    normalize=True  -> row-normalized (per true class)
    """
    set_paper_style()
    plt.figure(figsize=(6.2, 5.2))

    if normalize:
        cm_display = cm.astype(np.float64)
        row_sums = cm_display.sum(axis=1, keepdims=True) + 1e-12
        cm_display = cm_display / row_sums
        fmt = ".2f"
    else:
        # ✅ ensure integer counts for display + formatting
        cm_display = np.rint(cm).astype(int)
        fmt = "d"

    plt.imshow(cm_display, interpolation="nearest", cmap="Oranges")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_display.max() * 0.55 if cm_display.max() > 0 else 0.5

    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            plt.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                color="white" if val > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    _finalize_and_save(out_path, dpi=300)



# =========================================================
# ROC (OvR) + Macro AUC
# =========================================================
def plot_roc_ovr(y_true, y_prob, class_names, out_path):
    set_paper_style()
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(6.4, 4.8))

    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    # Macro-average AUC (simple mean of class AUCs)
    macro_auc = float(np.mean(aucs)) if len(aucs) else 0.0

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) | Macro AUC={macro_auc:.3f}")
    plt.grid(True, alpha=0.20)
    plt.legend(frameon=False, loc="lower right")
    _finalize_and_save(out_path, dpi=300)


# =========================================================
# RELIABILITY / CALIBRATION PLOT
# =========================================================
def plot_reliability_diagram(bin_conf, bin_acc, ece, out_path):
    """
    bin_conf: mean confidence per bin
    bin_acc:  accuracy per bin
    ece: Expected Calibration Error
    """
    set_paper_style()
    plt.figure(figsize=(6.0, 4.8))

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect Calibration")

    # Reliability curve
    plt.plot(bin_conf, bin_acc, marker="o", linewidth=2.0, label="Model")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.grid(True, alpha=0.20)
    plt.legend(frameon=False, loc="lower right")
    _finalize_and_save(out_path, dpi=300)
