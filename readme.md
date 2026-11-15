# PneumoniaMNIST Classification (HOG + SVM vs. CNN)

Binary classification of $28 \times 28$ grayscale pediatric chest X-ray images (**Normal** vs. **Pneumonia**) from the PneumoniaMNIST dataset. This project benchmarks a classical computer vision pipeline (**HOG + Linear SVM**) against a custom convolutional neural network (**`PneumoniaCNN`**) across multiple contrast-enhancement preprocessing techniques.

---

## Project Structure

```text
MIP-Final/
├── data/                        # Pre-split NumPy arrays (train, val, test)
│   ├── train_images.npy
│   ├── train_labels.npy
│   ├── val_images.npy
│   ├── val_labels.npy
│   ├── test_images.npy
│   └── test_labels.npy
├── notebooks/                   # Exploratory analysis & visualization notebooks
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_hog_exploration.ipynb
│   ├── 03_cnn_exploration.ipynb
│   └── 04_preprocessing.ipynb
├── outputs/
│   ├── figures/                 # Loss curves, confusion matrices, ROC/PR, error grids
│   └── results/                 # Saved model checkpoints (.pt) & metric logs (.txt)
├── src/
│   ├── preprocessing.py         # Grayscale preprocessing: original, clahe, hist_eq
│   ├── hog_svm.py               # HOG feature extraction + Linear SVM baseline
│   ├── cnn.py                   # PneumoniaCNN architecture (Conv-BN-ReLU + GAP)
│   ├── train_cnn.py             # CNN training, validation, and test evaluation CLI
│   └── visualize_results.py     # Generates comparison, ROC/PR, and error plots
├── cnn_tests.md                 # Full terminal logs for CNN preprocessing & tuning runs
└── requirements.txt             # Python dependencies