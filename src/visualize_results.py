import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.feature import hog
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnn import PneumoniaCNN
from src.preprocessing import preprocess

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def load_npy_split(split: str, method: str = "original"):
    """Load images and labels from data/ and apply src.preprocessing.preprocess."""
    images = np.load(DATA_DIR / f"{split}_images.npy")
    labels = np.load(DATA_DIR / f"{split}_labels.npy").reshape(-1).astype(np.int64)

    if images.ndim == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)
    if images.dtype != np.uint8:
        scale = 255.0 if images.max() <= 1.0 else 1.0
        images = np.clip(images * scale, 0, 255).astype(np.uint8)

    processed = np.stack([preprocess(img, method=method) for img in images])
    return processed, labels


@torch.no_grad()
def predict_cnn(ckpt_path: Path, images: np.ndarray, batch_size: int = 128):
    """Run PneumoniaCNN inference and return positive-class probabilities."""
    model = PneumoniaCNN().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    tensor_x = torch.from_numpy(images).unsqueeze(1).float() / 255.0
    probs = []
    for start in range(0, len(tensor_x), batch_size):
        batch = tensor_x[start : start + batch_size].to(DEVICE)
        logits = model(batch)
        batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(batch_probs)
    return np.concatenate(probs)


def get_svm_validation_results(val_images: np.ndarray, val_labels: np.ndarray):
    """Evaluate saved SVM baseline or parse outputs/results/svm_metrics.txt."""
    svm_model_path = RESULTS_DIR / "svm_baseline.joblib"
    svm_txt_path = RESULTS_DIR / "svm_metrics.txt"

    if svm_model_path.exists():
        svm_model = joblib.load(svm_model_path)
        features = np.stack(
            [
                hog(
                    img,
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                )
                for img in val_images
            ]
        )
        preds = svm_model.predict(features)
        scores = svm_model.decision_function(features)
        metrics = {
            "accuracy": accuracy_score(val_labels, preds),
            "precision": precision_score(val_labels, preds, zero_division=0),
            "recall": recall_score(val_labels, preds, zero_division=0),
            "f1": f1_score(val_labels, preds, zero_division=0),
        }
        return metrics, scores

    if svm_txt_path.exists():
        text = svm_txt_path.read_text()
        parsed = {}
        for key in ["accuracy", "precision", "recall", "f1"]:
            match = re.search(rf"{key}['\"]?\s*[:=]\s*([0-9.]+)", text, re.IGNORECASE)
            if match:
                parsed[key] = float(match.group(1))
        if len(parsed) == 4:
            return parsed, None

    return None, None


def plot_preprocessing_comparison():
    """1. Compare Original, CLAHE, and Hist Eq validation metrics & confusion matrices."""
    runs = {
        "Original\n(Baseline)": {
            "metrics": [0.9676, 0.9869, 0.9692, 0.9780],
            "cm": np.array([[130, 5], [12, 377]]),
        },
        "CLAHE\n(Best Val)": {
            "metrics": [0.9790, 0.9871, 0.9846, 0.9858],
            "cm": np.array([[130, 5], [6, 383]]),
        },
        "Hist Eq": {
            "metrics": [0.9733, 0.9747, 0.9897, 0.9821],
            "cm": np.array([[125, 10], [4, 385]]),
        },
    }
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)

    # Top row: Grouped metric bar chart
    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    for idx, (label, data) in enumerate(runs.items()):
        offset = (idx - 1) * width
        bars = ax_bar.bar(
            x + offset,
            data["metrics"],
            width,
            label=label.replace("\n", " "),
            color=colors[idx],
        )
        for bar in bars:
            h = bar.get_height()
            ax_bar.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax_bar.set_ylim(0.92, 1.005)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, fontsize=10)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("CNN Validation Performance by Preprocessing Method", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend(frameon=True, loc="lower right")

    # Bottom row: Side-by-side confusion matrices
    for idx, (label, data) in enumerate(runs.items()):
        ax_cm = fig.add_subplot(gs[1, idx])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=data["cm"], display_labels=["Normal", "Pneumonia"]
        )
        disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
        ax_cm.set_title(label.replace("\n", " "), fontsize=10.5)

    out_path = FIGURES_DIR / "cnn_preprocessing_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_roc_and_pr_curves(val_orig, val_clahe, val_labels, svm_scores):
    """2. Plot ROC and Precision-Recall curves across available models."""
    models = {}
    if svm_scores is not None:
        models["HOG + Linear SVM"] = svm_scores

    baseline_ckpt = RESULTS_DIR / "cnn_baseline.pt"
    if baseline_ckpt.exists():
        models["CNN (Original)"] = predict_cnn(baseline_ckpt, val_orig)

    clahe_ckpt = RESULTS_DIR / "cnn_clahe.pt"
    if clahe_ckpt.exists():
        models["CNN (CLAHE)"] = predict_cnn(clahe_ckpt, val_clahe)

    if not models:
        print("Skipping ROC/PR plot: no saved model checkpoints found.")
        return

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for name, scores in models.items():
        fpr, tpr, _ = roc_curve(val_labels, scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

        prec, rec, _ = precision_recall_curve(val_labels, scores)
        ap = average_precision_score(val_labels, scores)
        ax_pr.plot(rec, prec, lw=2, label=f"{name} (AP = {ap:.4f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax_roc.set_title("Validation ROC Curve", fontsize=11.5, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(True, linestyle="--", alpha=0.4)
    ax_roc.legend(loc="lower right")

    ax_pr.set_title("Validation Precision-Recall Curve", fontsize=11.5, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, linestyle="--", alpha=0.4)
    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    out_path = FIGURES_DIR / "cnn_roc_pr_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_svm_vs_cnn(svm_metrics):
    """3. Compare HOG+SVM baseline against CNN (Original) and CNN (CLAHE)."""
    if svm_metrics is None:
        print("Skipping SVM vs CNN plot: svm_metrics.txt / svm_baseline.joblib not found.")
        return

    model_scores = {
        "HOG + SVM (Baseline)": [
            svm_metrics["accuracy"],
            svm_metrics["precision"],
            svm_metrics["recall"],
            svm_metrics["f1"],
        ],
        "CNN (Original)": [0.9676, 0.9869, 0.9692, 0.9780],
        "CNN (CLAHE)": [0.9790, 0.9871, 0.9846, 0.9858],
    }
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ["#8C8C8C", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metric_names))
    width = 0.25

    for idx, (label, values) in enumerate(model_scores.items()):
        bars = ax.bar(x + (idx - 1) * width, values, width, label=label, color=colors[idx])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    min_val = min(min(v) for v in model_scores.values())
    ax.set_ylim(max(0.0, min_val - 0.05), 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Validation Score")
    ax.set_title("Model Benchmark: HOG + SVM vs. PneumoniaCNN", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = FIGURES_DIR / "cnn_vs_svm_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_error_analysis(val_clahe: np.ndarray, val_labels: np.ndarray):
    """4. Visualize False Positives and False Negatives on the validation set."""
    ckpt_path = RESULTS_DIR / "cnn_clahe.pt"
    if not ckpt_path.exists():
        ckpt_path = RESULTS_DIR / "cnn_baseline.pt"
    if not ckpt_path.exists():
        print("Skipping error analysis plot: no checkpoint found.")
        return

    probs = predict_cnn(ckpt_path, val_clahe)
    preds = (probs >= 0.5).astype(int)

    fp_idx = np.where((preds == 1) & (val_labels == 0))[0]
    fn_idx = np.where((preds == 0) & (val_labels == 1))[0]

    # Sort by highest confidence errors
    fp_idx = fp_idx[np.argsort(-probs[fp_idx])][:5]
    fn_idx = fn_idx[np.argsort(probs[fn_idx])][:5]

    cols = max(len(fp_idx), len(fn_idx), 1)
    fig, axes = plt.subplots(2, cols, figsize=(2.4 * cols, 5.2))
    if cols == 1:
        axes = axes.reshape(2, 1)

    for c in range(cols):
        # Row 0: False Positives
        ax_fp = axes[0, c]
        ax_fp.axis("off")
        if c < len(fp_idx):
            i = fp_idx[c]
            ax_fp.imshow(val_clahe[i], cmap="gray")
            ax_fp.set_title(f"FP (idx={i})\nP(Pneu)={probs[i]:.2f}", fontsize=9, color="#B22222")

        # Row 1: False Negatives
        ax_fn = axes[1, c]
        ax_fn.axis("off")
        if c < len(fn_idx):
            i = fn_idx[c]
            ax_fn.imshow(val_clahe[i], cmap="gray")
            ax_fn.set_title(f"FN (idx={i})\nP(Pneu)={probs[i]:.2f}", fontsize=9, color="#1F4E78")

    fig.suptitle(
        "Validation Error Analysis: Top False Positives (Top) & False Negatives (Bottom)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = FIGURES_DIR / "cnn_error_analysis.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_val_vs_test_summary():
    """5. Compare Validation vs. Test metrics and Test confusion matrix for tuned CLAHE run."""
    val_metrics = [0.9351, 0.9944, 0.9177, 0.9545]
    test_metrics = [0.9038, 0.8986, 0.9538, 0.9254]
    test_cm = np.array([[192, 42], [18, 372]])
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig, (ax_bar, ax_cm) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    x = np.arange(len(metric_names))
    width = 0.34
    b1 = ax_bar.bar(x - width / 2, val_metrics, width, label="Validation (n=524)", color="#4C72B0")
    b2 = ax_bar.bar(x + width / 2, test_metrics, width, label="Test (n=624)", color="#55A868")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax_bar.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax_bar.set_ylim(0.84, 1.02)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Tuned CNN (CLAHE + Weights + Aug): Val vs. Test", fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend(loc="lower right")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=test_cm, display_labels=["Normal (0)", "Pneumonia (1)"]
    )
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Test Confusion Matrix (CLAHE)", fontweight="bold")

    plt.tight_layout()
    out_path = FIGURES_DIR / "cnn_test_summary.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    val_orig, val_labels = load_npy_split("val", method="original")
    val_clahe, _ = load_npy_split("val", method="clahe")
    svm_metrics, svm_scores = get_svm_validation_results(val_orig, val_labels)

    plot_preprocessing_comparison()
    plot_roc_and_pr_curves(val_orig, val_clahe, val_labels, svm_scores)
    plot_svm_vs_cnn(svm_metrics)
    plot_error_analysis(val_clahe, val_labels)
    plot_val_vs_test_summary()


if __name__ == "__main__":
    main()