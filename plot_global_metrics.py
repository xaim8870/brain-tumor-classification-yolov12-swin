import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PATHS
# ===============================
CSV_PATH = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\figshare_class3\metrics_global.csv"
OUT_DIR  = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\figshare_class3\paper_plots\global"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ===============================
# PAPER STYLE (same as yours)
# ===============================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

TRAIN_COLOR = "#1f77b4"   # dark blue
VAL_COLOR   = "#ff7f0e"   # orange
LINE_W      = 1.8

# ===============================
# HELPERS
# ===============================
def col_exists_and_valid(col):
    return col in df.columns and not df[col].isna().all()

def style_legend():
    leg = plt.legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(1.0)

def save_plot(fname):
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300)
    plt.close()

def pct_label(name, series):
    return f"{name} (final = {series.iloc[-1] * 100:.2f}%)"

def loss_label(name, series):
    return f"{name} (final = {series.iloc[-1]:.3f})"

def raw_label(name, series):
    return f"{name} (final = {series.iloc[-1]:.6g})"

# ===============================
# 1) Accuracy + Loss (explicit)
# ===============================
if col_exists_and_valid("train_accuracy") and col_exists_and_valid("val_accuracy"):
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(df["epoch"], df["train_accuracy"] * 100,
             color=TRAIN_COLOR, linewidth=LINE_W,
             label=pct_label("Train", df["train_accuracy"]))
    plt.plot(df["epoch"], df["val_accuracy"] * 100,
             color=VAL_COLOR, linewidth=LINE_W,
             label=pct_label("Validation", df["val_accuracy"]))
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Training and Validation Accuracy")
    style_legend(); save_plot("accuracy.png")

if col_exists_and_valid("train_loss") and col_exists_and_valid("val_loss"):
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(df["epoch"], df["train_loss"],
             color=TRAIN_COLOR, linewidth=LINE_W,
             label=loss_label("Train", df["train_loss"]))
    plt.plot(df["epoch"], df["val_loss"],
             color=VAL_COLOR, linewidth=LINE_W,
             label=loss_label("Validation", df["val_loss"]))
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss"); plt.title("Training and Validation Loss")
    style_legend(); save_plot("loss.png")

# ===============================
# 2) Auto-plot every metric pair: train_* vs val_*
# ===============================
skip = {"train_loss", "val_loss", "train_accuracy", "val_accuracy"}  # already plotted

train_cols = [c for c in df.columns if c.startswith("train_") and c not in skip]
for train_col in train_cols:
    base = train_col.replace("train_", "", 1)
    val_col = f"val_{base}"
    if not col_exists_and_valid(val_col):
        continue

    # Decide scaling/labels
    is_prob_metric = any(k in base for k in ["precision", "recall", "f1", "auc"])
    is_loss_like   = any(k in base for k in ["nll", "loss"])

    plt.figure(figsize=(6.4, 4.0))
    if is_prob_metric:
        plt.plot(df["epoch"], df[train_col] * 100,
                 color=TRAIN_COLOR, linewidth=LINE_W,
                 label=pct_label("Train", df[train_col]))
        plt.plot(df["epoch"], df[val_col] * 100,
                 color=VAL_COLOR, linewidth=LINE_W,
                 label=pct_label("Validation", df[val_col]))
        plt.ylabel(f"{base} (%)")
    elif is_loss_like:
        plt.plot(df["epoch"], df[train_col],
                 color=TRAIN_COLOR, linewidth=LINE_W,
                 label=loss_label("Train", df[train_col]))
        plt.plot(df["epoch"], df[val_col],
                 color=VAL_COLOR, linewidth=LINE_W,
                 label=loss_label("Validation", df[val_col]))
        plt.ylabel(base)
    else:
        plt.plot(df["epoch"], df[train_col],
                 color=TRAIN_COLOR, linewidth=LINE_W,
                 label=raw_label("Train", df[train_col]))
        plt.plot(df["epoch"], df[val_col],
                 color=VAL_COLOR, linewidth=LINE_W,
                 label=raw_label("Validation", df[val_col]))
        plt.ylabel(base)

    title = base.replace("_", " ").title()
    plt.xlabel("Epoch")
    plt.title(title)
    style_legend()
    save_plot(f"{base}.png")

# ===============================
# 3) Learning Rate plot (if present)
# ===============================
if col_exists_and_valid("lr"):
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(df["epoch"], df["lr"], color="black", linewidth=1.6, label="LR")
    plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.title("Learning Rate Schedule")
    style_legend(); save_plot("lr.png")

print("✅ All valid global metric plots generated successfully.")
print("📁 Saved to:", OUT_DIR)
