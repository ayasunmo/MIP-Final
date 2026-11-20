
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "outputs" / "results" / "svm_week3"
FIGURES = ROOT / "outputs" / "figures" / "svm_week3"

FIGURES.mkdir(parents=True, exist_ok=True)


# 1. Load SVM results
svm = pd.read_csv(RESULTS / "validation_results.csv")


# 2. CNN validation confusion matrices
# Source: CNN Weekly Progress Report, Week 3
# Order: TN, FP, FN, TP

cnn_matrices = {
    "original": [130, 5, 12, 377],
    "clahe": [130, 5, 6, 383],
    "hist_eq": [125, 10, 4, 385]
}

cnn_results = []

for method, (tn, fp, fn, tp) in cnn_matrices.items():

    f1_positive = 2 * tp / (2 * tp + fp + fn)
    f1_negative = 2 * tn / (2 * tn + fp + fn)

    cnn_results.append({
        "model": "CNN",
        "preprocessing": method,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1": f1_positive,
        "macro_f1": (f1_positive + f1_negative) / 2,
        "specificity": tn / (tn + fp),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

cnn = pd.DataFrame(cnn_results)


# 3. Combine SVM and CNN results
comparison = pd.concat(
    [svm, cnn],
    ignore_index=True
)

comparison.to_csv(
    RESULTS / "svm_cnn_comparison.csv",
    index=False
)

print("SVM vs CNN validation results:")
print(comparison.to_string(index=False))


# 4. Prepare data for visualization
methods = ["original", "clahe", "hist_eq"]

names = [
    "Original",
    "CLAHE",
    "Hist. Eq."
]

svm = svm.set_index("preprocessing").loc[methods]
cnn = cnn.set_index("preprocessing").loc[methods]

metrics = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1-score"),
    ("macro_f1", "Macro-F1"),
    ("specificity", "Specificity")
]


# 5. Create comparison plots
fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 10)
)

x = np.arange(len(methods))
width = 0.35

for ax, (metric, title) in zip(axes.ravel(), metrics):

    bars_svm = ax.bar(
        x - width / 2,
        svm[metric] * 100,
        width,
        label="HOG + SVM",
        color="#2563eb"
    )

    bars_cnn = ax.bar(
        x + width / 2,
        cnn[metric] * 100,
        width,
        label="CNN",
        color="#16a34a"
    )

    ax.bar_label(bars_svm, fmt="%.1f", fontsize=9)
    ax.bar_label(bars_cnn, fmt="%.1f", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)

    ax.set_title(title)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.2)

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2
)

fig.suptitle(
    "Traditional Machine Learning vs Deep Learning\n"
    "PneumoniaMNIST Validation Performance",
    fontsize=17
)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])

plt.savefig(
    FIGURES / "svm_vs_cnn_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison completed successfully!")
print("CSV saved:", RESULTS / "svm_cnn_comparison.csv")
print("Figure saved:", FIGURES / "svm_vs_cnn_comparison.png")
