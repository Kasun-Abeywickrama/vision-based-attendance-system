from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def resize_to_width(image: np.ndarray, width: int = 1400) -> np.ndarray:
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(
        image,
        (width, int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)