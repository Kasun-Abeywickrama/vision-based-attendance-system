import numpy as np


def extract_signature_roi(image: np.ndarray, x_left: int, x_right: int,
                           y_top: int, y_bottom: int) -> np.ndarray:
    """Crop the signature cell for a single student row."""
    return image[y_top:y_bottom, x_left:x_right]
