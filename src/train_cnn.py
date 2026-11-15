import argparse
import copy
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow running both `python src/train_cnn.py` and `python -m src.train_cnn`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cnn import PneumoniaCNN
from src.preprocessing import preprocess


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NpyXrayDataset(Dataset):
    """Loads .npy X-ray arrays and applies src.preprocessing.preprocess."""

    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        method: str = "original",
        transform=None,
    ):
        raw_images = np.load(images_path)
        raw_labels = np.load(labels_path).reshape(-1).astype(np.int64)

        # Ensure 2D uint8 arrays as expected by src.preprocessing.preprocess
        if raw_images.ndim == 4 and raw_images.shape[-1] == 1:
            raw_images = raw_images.squeeze(-1)
        if raw_images.dtype != np.uint8:
            scale = 255.0 if raw_images.max() <= 1.0 else 1.0
            raw_images = np.clip(raw_images * scale, 0, 255).astype(np.uint8)

        self.images = np.stack([preprocess(img, method=method) for img in raw_images])
        self.labels = raw_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]  # (28, 28) uint8
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train PneumoniaCNN on data/*.npy")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--preprocess",
        type=str,
        default="original",
        choices=["original", "clahe", "hist_eq"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    results_dir = PROJECT_ROOT / "outputs" / "results"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_transform = transforms.ToTensor()
    if args.augment:
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = eval_transform

    train_ds = NpyXrayDataset(
        args.data_dir / "train_images.npy",
        args.data_dir / "train_labels.npy",
        method=args.preprocess,
        transform=train_transform,
    )
    val_ds = NpyXrayDataset(
        args.data_dir / "val_images.npy",
        args.data_dir / "val_labels.npy",
        method=args.preprocess,
        transform=eval_transform,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Inverse-frequency class weights to counter PneumoniaMNIST class imbalance
    if args.use_class_weights:
        counts = np.bincount(train_ds.labels, minlength=2).astype(np.float32)
        weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    model = PneumoniaCNN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    train_losses, val_losses = [], []
    best_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

    model.load_state_dict(best_state)
    ckpt_name = (
        "cnn_baseline.pt"
        if args.preprocess == "original" and not args.use_class_weights and not args.augment
        else f"cnn_{args.preprocess}.pt"
    )
    torch.save(best_state, results_dir / ckpt_name)

    val_metrics = evaluate(model, val_loader, criterion, device)
    report_lines = [
        f"Split: Validation | Preprocess: {args.preprocess} | "
        f"ClassWeights: {args.use_class_weights} | Augment: {args.augment}",
        f"Accuracy:  {val_metrics['accuracy']:.4f}",
        f"Precision: {val_metrics['precision']:.4f}",
        f"Recall:    {val_metrics['recall']:.4f}",
        f"F1:        {val_metrics['f1']:.4f}",
        "Confusion matrix:",
        str(val_metrics["cm"]),
    ]

    if args.eval_test:
        test_ds = NpyXrayDataset(
            args.data_dir / "test_images.npy",
            args.data_dir / "test_labels.npy",
            method=args.preprocess,
            transform=eval_transform,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        test_metrics = evaluate(model, test_loader, criterion, device)
        report_lines.extend(
            [
                "",
                f"Split: Test | Preprocess: {args.preprocess}",
                f"Accuracy:  {test_metrics['accuracy']:.4f}",
                f"Precision: {test_metrics['precision']:.4f}",
                f"Recall:    {test_metrics['recall']:.4f}",
                f"F1:        {test_metrics['f1']:.4f}",
                "Confusion matrix:",
                str(test_metrics["cm"]),
            ]
        )
        cm_to_plot = test_metrics["cm"]
        cm_title = f"CNN Test Confusion Matrix ({args.preprocess})"
    else:
        cm_to_plot = val_metrics["cm"]
        cm_title = f"CNN Validation Confusion Matrix ({args.preprocess})"

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (results_dir / "cnn_metrics.txt").write_text(report_text + "\n")

    # Training curves
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "cnn_loss.png", dpi=200)
    plt.close()

    # Confusion matrix figure (matches outputs/figures/svm_confusion_matrix.png)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_to_plot, display_labels=["Normal (0)", "Pneumonia (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(cm_title)
    plt.tight_layout()
    plt.savefig(figures_dir / "cnn_confusion_matrix.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()