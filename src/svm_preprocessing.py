
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

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

from src.preprocessing import preprocess


# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "results" / "svm_week3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the shared dataset
X_train = np.load(DATA_DIR / "train_images.npy")
y_train = np.load(DATA_DIR / "train_labels.npy").ravel()

X_val = np.load(DATA_DIR / "val_images.npy")
y_val = np.load(DATA_DIR / "val_labels.npy").ravel()

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)


# 3. Preprocessing and HOG feature extraction
def extract_features(images, method):
    features = []

    for image in images:
        processed = preprocess(image, method)

        feature = hog(
            processed,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features.append(feature)

    return np.array(features)


# 4. Experimental configurations
methods = ["original", "clahe", "hist_eq"]
results = []


# 5. Train and evaluate each model
for method in methods:

    print("\n" + "=" * 50)
    print("Experiment:", method.upper())
    print("=" * 50)

    print("Extracting HOG features...")

    X_train_hog = extract_features(X_train, method)
    X_val_hog = extract_features(X_val, method)

    print("Train features:", X_train_hog.shape)
    print("Validation features:", X_val_hog.shape)

    assert np.isfinite(X_train_hog).all()
    assert np.isfinite(X_val_hog).all()

    # Create a new SVM model
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=1.0)
    )

    print("Training SVM...")

    model.fit(X_train_hog, y_train)

    # Validation predictions
    y_pred = model.predict(X_val_hog)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Evaluation metrics
    metrics = {
        "model": "HOG + SVM",
        "preprocessing": method,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, y_pred,
            average="macro",
            zero_division=0
        ),
        "specificity": tn / (tn + fp),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    results.append(metrics)

    # Print results
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Confusion matrix:")
    print(cm)

    # Save trained model
    joblib.dump(
        model,
        OUTPUT_DIR / f"svm_{method}.joblib"
    )

    # Save confusion matrix
    np.save(
        OUTPUT_DIR / f"cm_{method}.npy",
        cm
    )

    # Save validation predictions
    pd.DataFrame({
        "image_index": np.arange(len(y_val)),
        "true_label": y_val,
        "predicted_label": y_pred
    }).to_csv(
        OUTPUT_DIR / f"predictions_{method}.csv",
        index=False
    )

    print("Model and results saved!")


# 6. Save comparison table
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "validation_results.csv",
    index=False
)

print("\nFINAL VALIDATION RESULTS")
print(results_df.to_string(index=False))

print("\nAll experiments completed!")
print("Results saved to:", OUTPUT_DIR)
