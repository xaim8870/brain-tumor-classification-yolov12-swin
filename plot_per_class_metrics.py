import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATHS
# ===============================
CSV_PATH = r"D:\Brain-Tumor (2)\brain_tumor_classification\Output-results\Output-results\metrics_per_class.csv"
OUT_DIR = r"D:\Brain-Tumor (2)\brain_tumor_classification\outputs\paper_plots\per_class"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ===============================
# PAPER STYLE
# ===============================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

LINE_W = 1.8

# Consistent class colors (paper friendly)
CLASS_COLORS = {
    "glioma": "#1f77b4",        # blue
    "meningioma": "#ff7f0e",    # orange
    "no_tumor": "#2ca02c",      # green
    "pituitary": "#d62728"      # red
}

# ===============================
# HELPERS
# ===============================
def final_pct(series):
    return f"{series.iloc[-1] * 100:.2f}%"


def style_legend():
    leg = plt.legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(1.0)


# ===============================
# GENERIC METRIC PLOTTER
# ===============================
def plot_metric(metric_col, title, ylabel, filename):
    plt.figure(figsize=(6.6, 4.2))

    for cls in sorted(df["class_name"].unique()):
        cls_df = df[df["class_name"] == cls].sort_values("epoch")

        plt.plot(
            cls_df["epoch"],
            cls_df[metric_col] * 100,
            linewidth=LINE_W,
            color=CLASS_COLORS[cls],
            label=f"{cls} (final = {final_pct(cls_df[metric_col])})"
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    style_legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Saved: {out_path}")


# ===============================
# PLOTS
# ===============================
plot_metric(
    metric_col="precision",
    title="Per-Class Precision over Training",
    ylabel="Precision (%)",
    filename="precision_per_class.png"
)

plot_metric(
    metric_col="recall_sensitivity",
    title="Per-Class Recall (Sensitivity) over Training",
    ylabel="Recall (%)",
    filename="recall_per_class.png"
)

plot_metric(
    metric_col="f1",
    title="Per-Class F1-score over Training",
    ylabel="F1-score (%)",
    filename="f1_per_class.png"
)

plot_metric(
    metric_col="specificity",
    title="Per-Class Specificity over Training",
    ylabel="Specificity (%)",
    filename="specificity_per_class.png"
)

print("\n✅ All per-class metric plots generated (paper-quality).")
