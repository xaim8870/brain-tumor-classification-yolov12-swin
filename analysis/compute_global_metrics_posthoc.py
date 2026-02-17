import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    log_loss,
    roc_auc_score
)

# ===============================
# PATH
# ===============================
NPZ_PATH = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\val_best_predictions.npz"
OUT_CSV = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\global_metrics_final.csv"

data = np.load(NPZ_PATH)
y_true = data["y_true"]
y_pred = data["y_pred"]
y_prob = data["y_prob"]

# ===============================
# METRICS
# ===============================
acc = accuracy_score(y_true, y_pred)

p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro"
)

p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted"
)

auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")

nll = log_loss(y_true, y_prob)

# ===============================
# SAVE
# ===============================
df = pd.DataFrame([{
    "accuracy": acc,
    "precision_macro": p_macro,
    "recall_macro": r_macro,
    "f1_macro": f1_macro,
    "precision_weighted": p_weight,
    "recall_weighted": r_weight,
    "f1_weighted": f1_weight,
    "auc_macro_ovr": auc_macro,
    "auc_weighted_ovr": auc_weighted,
    "nll": nll
}])

df.to_csv(OUT_CSV, index=False)

print("✅ Global metrics computed without retraining")
print(df.round(4))
