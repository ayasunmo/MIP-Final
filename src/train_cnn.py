
from pathlib import Path
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix
)

from src.cnn import PneumoniaCNN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

Path("outputs/results").mkdir(parents=True, exist_ok=True)
Path("outputs/figures").mkdir(parents=True, exist_ok=True)

transform = transforms.ToTensor()

train_ds = PneumoniaMNIST(
    split="train", transform=transform,
    download=True, size=28
)

val_ds = PneumoniaMNIST(
    split="val", transform=transform,
    download=True, size=28
)

train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True
)
val_loader = DataLoader(
    val_ds, batch_size=64, shuffle=False
)

model = PneumoniaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

train_losses = []
val_losses = []
best_loss = float("inf")
best_state = None

for epoch in range(15):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_ds)

    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)

    val_loss = running_val_loss / len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch+1:02d}/15 | "
        f"Train: {train_loss:.4f} | "
        f"Val: {val_loss:.4f}"
    )

model.load_state_dict(best_state)

torch.save(
    best_state, "outputs/results/cnn_baseline.pt"
)

# Validation predictions
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.view(-1).numpy().tolist())

print("Validation metrics:")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

# Training curves
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/cnn_loss.png", dpi=200)
plt.show()
