
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "outputs" / "results" / "svm_week3"
FIGURES = ROOT / "outputs" / "figures" / "svm_week3"

FIGURES.mkdir(parents=True, exist_ok=True)

# Load saved experimental results
df = pd.read_csv(RESULTS / "validation_results.csv")

methods = ["original", "clahe", "hist_eq"]
names = ["Original", "CLAHE", "Histogram Equalization"]

df = df.set_index("preprocessing").loc[methods].reset_index()


# 1. Metrics comparison
metrics = [
    "accuracy", "precision", "recall",
    "f1", "macro_f1", "specificity"
]

metric_names = [
    "Accuracy", "Precision", "Recall",
    "F1-score", "Macro-F1", "Specificity"
]

x = np.arange(len(metrics))
width = 0.25

colors = ["#2563eb", "#f59e0b", "#16a34a"]

fig, ax = plt.subplots(figsize=(14, 6))

for i, method in enumerate(methods):

    values = df.loc[i, metrics].to_numpy(dtype=float) * 100

    bars = ax.bar(
        x + (i - 1) * width,
        values,
        width,
        label=names[i],
        color=colors[i]
    )

    ax.bar_label(bars, fmt="%.1f", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metric_names)

ax.set_ylabel("Score (%)")
ax.set_ylim(0, 108)
ax.set_title("HOG + SVM: Preprocessing Comparison")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    FIGURES / "svm_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 2. Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, method in enumerate(methods):

    cm = np.load(RESULTS / f"cm_{method}.npy")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Pneumonia"]
    )

    disp.plot(
        ax=axes[i],
        cmap="Blues",
        colorbar=False
    )

    disp.im_.set_clim(0, 389)

    axes[i].set_title(names[i])

fig.suptitle(
    "HOG + SVM: Validation Confusion Matrices",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    FIGURES / "svm_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 3. False negatives and false positives
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(methods))
width = 0.35

fn_bars = ax.bar(
    x - width / 2,
    df["FN"],
    width,
    label="False Negatives",
    color="#dc2626"
)

fp_bars = ax.bar(
    x + width / 2,
    df["FP"],
    width,
    label="False Positives",
    color="#2563eb"
)

ax.bar_label(fn_bars)
ax.bar_label(fp_bars)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Number of Errors")
ax.set_title("HOG + SVM: Classification Error Analysis")
ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    FIGURES / "svm_error_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("All figures generated successfully!")
print("Saved to:", FIGURES)

for path in sorted(FIGURES.glob("*.png")):
    print(path.name)
