
import cv2
import numpy as np


def preprocess(image, method="original"):
    """Preprocess one grayscale uint8 X-ray image."""
    image = np.asarray(image)

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 image")

    if method == "original":
        return image.copy()

    if method == "clahe":
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4)
        )
        return clahe.apply(image)

    if method == "hist_eq":
        return cv2.equalizeHist(image)

    raise ValueError(f"Unknown method: {method}")
