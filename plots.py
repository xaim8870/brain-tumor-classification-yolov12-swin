def plot_confusion_matrix(cm, class_names, out_path, normalize=False):
    set_paper_style()
    plt.figure(figsize=(6.2, 5.2))

    cm_display = cm.astype(np.float64)

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True) + 1e-12
        cm_display = cm_display / row_sums
        fmt = ".2f"
    else:
        # ✅ force integer display for counts
        cm_display = np.rint(cm_display).astype(int)
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
            plt.text(j, i, format(val, fmt),
                     ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    _finalize_and_save(out_path, dpi=300)
