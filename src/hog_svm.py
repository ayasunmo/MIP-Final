from pathlib import Path
import numpy as np
from medmnist import PneumoniaMNIST
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import joblib


def extract_features(images):
    return np.stack([
        hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        for image in images
    ])


train = PneumoniaMNIST(
    split="train", download=True, size=28
)
val = PneumoniaMNIST(
    split="val", download=True, size=28
)

X_train = extract_features(train.imgs)
X_val = extract_features(val.imgs)

y_train = train.labels.ravel()
y_val = val.labels.ravel()

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1.0)
)

model.fit(X_train, y_train)
predictions = model.predict(X_val)

metrics = {
    "accuracy": accuracy_score(y_val, predictions),
    "precision": precision_score(
        y_val, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, predictions, zero_division=0
    ),
}

print("Validation metrics:", metrics)
print("Confusion matrix:")
print(confusion_matrix(y_val, predictions))

Path("outputs/results").mkdir(parents=True, exist_ok=True)
joblib.dump(model, "outputs/results/svm_baseline.joblib")
