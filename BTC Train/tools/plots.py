import os
import numpy as np
import pandas as pd

from utils.plots import plot_accuracy, plot_loss, plot_confusion_matrix

OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\figshare_class3"  # change if needed

global_csv_path = os.path.join(OUT_DIR, "metrics_global.csv")
per_class_csv_path = os.path.join(OUT_DIR, "metrics_per_class.csv")
preds_dir = os.path.join(OUT_DIR, "preds")
plot_dir = os.path.join(OUT_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

# 1) Curves
g = pd.read_csv(global_csv_path)
plot_accuracy(g, os.path.join(plot_dir, "accuracy_train_val.png"))
plot_loss(g, os.path.join(plot_dir, "loss_train_val.png"))

# 2) Best epoch confusion matrix (from saved preds)
best_epoch = int(g.loc[g["val_f1_macro"].idxmax(), "epoch"])
npz_path = os.path.join(preds_dir, f"val_epoch_{best_epoch:03d}.npz")

data = np.load(npz_path)
y_true = data["y_true"]
y_pred = data["y_pred"]

# Load class names saved by training
class_names_path = os.path.join(OUT_DIR, "class_names.json")
if os.path.exists(class_names_path):
    import json
    class_names = json.load(open(class_names_path, "r"))
else:
    # fallback: infer from per-class file
    pc = pd.read_csv(per_class_csv_path)
    class_names = sorted(pc["class_name"].unique().tolist())

# Build confusion matrix
num_classes = len(class_names)
cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[int(t), int(p)] += 1

plot_confusion_matrix(cm, class_names, os.path.join(plot_dir, f"cm_best_epoch_{best_epoch:03d}_counts.png"), normalize=False)
plot_confusion_matrix(cm, class_names, os.path.join(plot_dir, f"cm_best_epoch_{best_epoch:03d}_normalized.png"), normalize=True)

pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    os.path.join(plot_dir, f"cm_best_epoch_{best_epoch:03d}.csv")
)

print("✅ Visualisations created in:", plot_dir)
print("✅ Best epoch:", best_epoch)
