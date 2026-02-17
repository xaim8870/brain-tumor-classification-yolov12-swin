import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# PATHS
# ============================
GLOBAL_CSV = r"D:\Brain-Tumor (2)\brain_tumor_classification\new\new\metrics_global.csv"
PER_CLASS_CSV = r"D:\Brain-Tumor (2)\brain_tumor_classification\new\new\metrics_per_class.csv"

OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\paper_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# STYLE (PAPER QUALITY)
# ============================
sns.set_theme(style="whitegrid", font="serif", font_scale=1.3)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ============================
# LOAD DATA
# ============================
df_global = pd.read_csv(GLOBAL_CSV)
df_pc = pd.read_csv(PER_CLASS_CSV)

# ============================
# 1️⃣ ACCURACY (TRAIN vs VAL)
# ============================
plt.figure(figsize=(8, 5))
plt.plot(df_global["epoch"], df_global["train_accuracy"],
         label="Train Accuracy", linewidth=2)
plt.plot(df_global["epoch"], df_global["val_accuracy"],
         label="Validation Accuracy", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "accuracy_train_val.png"))
plt.close()

# ============================
# 2️⃣ LOSS (TRAIN vs VAL)
# ============================
plt.figure(figsize=(8, 5))
plt.plot(df_global["epoch"], df_global["train_loss"],
         label="Train Loss", linewidth=2)
plt.plot(df_global["epoch"], df_global["val_loss"],
         label="Validation Loss", linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_train_val.png"))
plt.close()

# ============================
# 3️⃣ MACRO F1
# ============================
plt.figure(figsize=(8, 5))
plt.plot(df_global["epoch"], df_global["val_f1_macro"],
         label="Macro F1", linewidth=2, color="green")

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation Macro F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "f1_macro.png"))
plt.close()

# ============================
# 4️⃣ OPTIONAL ROC-AUC (ONLY IF PRESENT)
# ============================
if "val_auc_macro_ovr" in df_global.columns:
    plt.figure(figsize=(8, 5))

    if "val_auc_macro_ovr" in df_global:
        plt.plot(df_global["epoch"], df_global["val_auc_macro_ovr"],
                 label="Macro AUC (OvR)", linewidth=2)

    if "val_auc_weighted_ovr" in df_global:
        plt.plot(df_global["epoch"], df_global["val_auc_weighted_ovr"],
                 label="Weighted AUC (OvR)", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Validation ROC-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_auc.png"))
    plt.close()

# ============================
# PER-CLASS PLOTS
# ============================
classes = df_pc["class_name"].unique()
palette = sns.color_palette("tab10", len(classes))

# ============================
# 5️⃣ PER-CLASS F1
# ============================
plt.figure(figsize=(9, 6))
for cls, color in zip(classes, palette):
    sub = df_pc[df_pc["class_name"] == cls]
    plt.plot(sub["epoch"], sub["f1"], label=cls, linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Per-Class F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_f1.png"))
plt.close()

# ============================
# 6️⃣ PER-CLASS RECALL (SENSITIVITY)
# ============================
plt.figure(figsize=(9, 6))
for cls, color in zip(classes, palette):
    sub = df_pc[df_pc["class_name"] == cls]
    plt.plot(sub["epoch"], sub["recall_sensitivity"], label=cls, linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Recall (Sensitivity)")
plt.title("Per-Class Sensitivity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_recall.png"))
plt.close()

# ============================
# 7️⃣ PER-CLASS PRECISION
# ============================
plt.figure(figsize=(9, 6))
for cls, color in zip(classes, palette):
    sub = df_pc[df_pc["class_name"] == cls]
    plt.plot(sub["epoch"], sub["precision"], label=cls, linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Per-Class Precision")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_precision.png"))
plt.close()

print("✅ Paper-quality plots generated successfully")
print("📁 Output directory:", OUT_DIR)
