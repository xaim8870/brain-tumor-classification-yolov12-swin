# training/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    log_loss
)

# =========================================================
# GLOBAL METRICS
# =========================================================
def compute_global_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    auc_macro = None
    auc_weighted = None
    nll = None

    if y_prob is not None:
        try:
            auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            pass

        try:
            nll = float(log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1]))))
        except Exception:
            nll = None

    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weight),
        "recall_weighted": float(r_weight),
        "f1_weighted": float(f1_weight),
        "auc_macro_ovr": None if auc_macro is None else float(auc_macro),
        "auc_weighted_ovr": None if auc_weighted is None else float(auc_weighted),
        "nll": None if nll is None else float(nll),
    }


# =========================================================
# PER-CLASS METRICS + CONFUSION MATRIX
# =========================================================
def compute_per_class_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    results = []

    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # sensitivity
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "class_id": c,
            "precision": float(precision),
            "recall_sensitivity": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
        })

    return results, cm


# =========================================================
# CALIBRATION (ECE + RELIABILITY BINS)
# =========================================================
def calibration_bins(y_true, y_prob, n_bins=15):
    """
    Returns:
      bin_conf: mean confidence per bin
      bin_acc : accuracy per bin
      ece     : expected calibration error
    Multi-class: confidence = max prob; prediction = argmax prob
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    correct = (pred == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = []
    bin_acc = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue

        c_mean = float(conf[mask].mean())
        a_mean = float(correct[mask].mean())
        bin_conf.append(c_mean)
        bin_acc.append(a_mean)

        ece += (mask.sum() / len(conf)) * abs(a_mean - c_mean)

    return np.array(bin_conf, dtype=np.float32), np.array(bin_acc, dtype=np.float32), float(ece)
