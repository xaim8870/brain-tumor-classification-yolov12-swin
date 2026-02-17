# training/train.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm

from utils.plots import plot_confusion_matrix

from training.metrics import compute_global_metrics, compute_per_class_metrics


def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
    device="cuda",
    use_amp=True,
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * imgs.size(0)

        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )

def save_confusion_matrix(cm, class_names, out_png, out_csv):
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(out_csv, index=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Best Epoch)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    class_names,
    out_dir,
    epochs=100,
    device="cuda",
    use_amp=True,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    scaler = GradScaler("cuda",enabled=use_amp)

    global_rows = []
    per_class_rows = []

    best_f1 = -1.0
    best_cm = None
    best_epoch = -1

    best_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        train_loss, tr_y, tr_p, tr_prob = run_one_epoch(
            model, train_loader, criterion,
            optimizer, scaler, device, use_amp
        )

        val_loss, va_y, va_p, va_prob = run_one_epoch(
            model, val_loader, criterion,
            None, None, device, use_amp
        )

        scheduler.step()

        train_g = compute_global_metrics(tr_y, tr_p, tr_prob)
        val_g = compute_global_metrics(va_y, va_p, va_prob)
        val_pc, cm = compute_per_class_metrics(va_y, va_p, len(class_names))

        # -------- GLOBAL CSV (FULL METRICS) --------
        global_rows.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_g.items()},
            **{f"val_{k}": v for k, v in val_g.items()},
        })

        # -------- PER CLASS CSV --------
        for m in val_pc:
            per_class_rows.append({
                "epoch": epoch,
                "class_id": m["class_id"],
                "class_name": class_names[m["class_id"]],
                "precision": m["precision"],
                "recall_sensitivity": m["recall_sensitivity"],
                "specificity": m["specificity"],
                "f1": m["f1"],
            })

        pd.DataFrame(global_rows).to_csv(
            os.path.join(out_dir, "metrics_global.csv"), index=False
        )
        pd.DataFrame(per_class_rows).to_csv(
            os.path.join(out_dir, "metrics_per_class.csv"), index=False
        )

        # Save predictions
        np.savez(
            os.path.join(preds_dir, f"val_epoch_{epoch:03d}.npz"),
            y_true=va_y, y_pred=va_p, y_prob=va_prob
        )

        if val_g["f1_macro"] > best_f1:
            best_f1 = val_g["f1_macro"]
            best_cm = cm.copy()
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                best_path
            )

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"val_acc={val_g['accuracy']:.4f} "
            f"val_f1_macro={val_g['f1_macro']:.4f}"
        )
    if best_cm is not None:
        plot_confusion_matrix(
            best_cm, class_names,
            os.path.join(out_dir, f"cm_best_epoch_{best_epoch:03d}_counts.png"),
            normalize=False
        )
        plot_confusion_matrix(
            best_cm, class_names,
            os.path.join(out_dir, f"cm_best_epoch_{best_epoch:03d}_normalized.png"),
            normalize=True
        )
        pd.DataFrame(best_cm, index=class_names, columns=class_names).to_csv(
            os.path.join(out_dir, f"cm_best_epoch_{best_epoch:03d}.csv"),
            index=True
        )


    return best_path
