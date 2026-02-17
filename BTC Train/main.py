import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.seed import set_seed
from utils.augmentations import train_transforms, val_transforms
from datasets.brain_tumor_dataset import BrainTumorDataset
from models.yolov12_swin_classifier import YOLOv12SwinClassifier
from training.losses import build_weighted_ce
from training.train import train_model
from training.metrics import compute_global_metrics, compute_per_class_metrics, calibration_bins
from utils.plots import (
    plot_accuracy, plot_loss,
    plot_confusion_matrix, plot_roc_ovr,
    plot_reliability_diagram
)

# ✅ FIGSHARE 3-CLASS DATASET ROOT
DATASET_ROOT = r"D:\Brain-Tumor (2)\brain_tumor_classification\data\Figshare-2\Figshare-2\dataset"
OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\figshare_class3"

BATCH_SIZE = 4
EPOCHS = 100


def evaluate_model(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_pred, y_prob


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Figshare is already 3-class, but keep exclude just in case
    EXCLUDE = ["no_tumor", "No_Tumor"]

    train_root = os.path.join(DATASET_ROOT, "train", "images")
    val_root   = os.path.join(DATASET_ROOT, "val", "images")
    test_root  = os.path.join(DATASET_ROOT, "test", "images")

    # optional: normalize folder names -> lowercase labels
    train_ds = BrainTumorDataset(train_root, transform=train_transforms(),
                                 exclude_classes=EXCLUDE, class_name_fn=str.lower)
    val_ds   = BrainTumorDataset(val_root,   transform=val_transforms(),
                                 exclude_classes=EXCLUDE, class_name_fn=str.lower)
    test_ds  = BrainTumorDataset(test_root,  transform=val_transforms(),
                                 exclude_classes=EXCLUDE, class_name_fn=str.lower)

    class_names = train_ds.classes
    print("Class names:", class_names)

    # ---------- class counts ----------
    counts = np.bincount([y for _, y in train_ds.samples], minlength=len(class_names)).tolist()
    print("Train class counts:", counts)

    # =========================================================
    # ✅ Imbalance handling (choose ONE primary method)
    # Method A (recommended): Weighted CE only
    # Method B: WeightedRandomSampler (if imbalance is strong)
    # =========================================================

    USE_SAMPLER = False  # set False if you want only weighted CE
    criterion = build_weighted_ce(counts, device=device, label_smoothing=0.02)
    if USE_SAMPLER:
        class_counts = np.bincount([y for _, y in train_ds.samples], minlength=len(class_names))
        class_weights = 1.0 / (class_counts + 1e-12)
        sample_weights = [float(class_weights[y]) for _, y in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = YOLOv12SwinClassifier(num_classes=len(class_names)).to(device)

    # ✅ If using sampler, you can still keep weighted CE (mild) OR switch to unweighted.
    # Keeping weighted CE is okay, but if you see instability, set weights off or lower smoothing.
    criterion = build_weighted_ce(counts, device=device, label_smoothing=0.02)

    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        (backbone_params if "backbone" in n else head_params).append(p)

    optimizer = AdamW(
        [
            {"params": head_params, "lr": 1e-4, "weight_decay": 5e-2},
            {"params": backbone_params, "lr": 1e-5, "weight_decay": 5e-2},
        ]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        class_names=class_names,
        out_dir=OUT_DIR,
        epochs=EPOCHS,
        device=device,
        use_amp=True
    )

    # ----- curves -----
    plot_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    global_csv = pd.read_csv(os.path.join(OUT_DIR, "metrics_global.csv"))
    plot_accuracy(global_csv, os.path.join(plot_dir, "accuracy_train_val.png"))
    plot_loss(global_csv, os.path.join(plot_dir, "loss_train_val.png"))

    # ----- eval best -----
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    summary = {"best_checkpoint": best_path}

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        y_true, y_pred, y_prob = evaluate_model(model, loader, device)

        g = compute_global_metrics(y_true, y_pred, y_prob)
        per_cls, cm = compute_per_class_metrics(y_true, y_pred, num_classes=len(class_names))
        bin_conf, bin_acc, ece = calibration_bins(y_true, y_prob, n_bins=15)

        summary[split_name] = {"global": g, "ece": float(ece), "per_class": per_cls}

        np.savez(os.path.join(OUT_DIR, f"{split_name}_best_predictions.npz"),
                 y_true=y_true, y_pred=y_pred, y_prob=y_prob)

        plot_confusion_matrix(cm, class_names, os.path.join(plot_dir, f"{split_name}_cm_counts.png"), normalize=False)
        plot_confusion_matrix(cm, class_names, os.path.join(plot_dir, f"{split_name}_cm_normalized.png"), normalize=True)
        plot_roc_ovr(y_true, y_prob, class_names, os.path.join(plot_dir, f"{split_name}_roc_ovr.png"))
        plot_reliability_diagram(bin_conf, bin_acc, ece, os.path.join(plot_dir, f"{split_name}_reliability.png"))

        print(f"\n[{split_name.upper()} BEST] acc={g['accuracy']:.4f} f1_macro={g['f1_macro']:.4f} ece={ece:.4f}")

    with open(os.path.join(OUT_DIR, "report_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Training + evaluation complete.")
    print("Best checkpoint:", best_path)
    print("Plots saved to:", plot_dir)



if __name__ == "__main__":
    main()
