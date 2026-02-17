import os, numpy as np
OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\figshare_class3"
PREDS_DIR = os.path.join(OUT_DIR, "preds")
npz_path = os.path.join(PREDS_DIR, "val_epoch_010.npz")  # change if needed

d = np.load(npz_path)
y_true = d["y_true"].astype(int)
y_pred = d["y_pred"].astype(int)

wrong_idx = np.where(y_true != y_pred)[0]
print("Total samples:", len(y_true))
print("Wrong predictions:", len(wrong_idx))
print("Accuracy:", 1 - (len(wrong_idx) / len(y_true)))

print("NPZ keys:", list(d.keys()))
if len(wrong_idx) > 0:
    print("First 20 wrong indices:", wrong_idx[:20])
    print("true->pred (first 20 wrong):", list(zip(y_true[wrong_idx[:20]], y_pred[wrong_idx[:20]])))
