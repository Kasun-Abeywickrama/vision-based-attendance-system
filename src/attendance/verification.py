from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .models import DetectionResult, Student

@dataclass(slots=True)
class VisionConfig:
    work_width: int = 1400
    dark_threshold: int = 175
    presence_ink_ratio: float = 0.020
    uncertain_band: float = 0.006
    save_intermediate: bool = True

class SheetDetectionError(RuntimeError):
    pass


def _rotate_same_size(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)



class AttendanceVision:
    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()

    def _deskew(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = gray[int(0.20*h):int(0.60*h), int(0.04*w):int(0.90*w)]
        edges = cv2.Canny(roi, 40, 120)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/360, threshold=80,
            minLineLength=max(100, int(0.22*w)), maxLineGap=30,
        )
        angles: list[float] = []
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                angle = math.degrees(math.atan2(y2-y1, x2-x1))
                if abs(angle) < 10:
                    angles.append(angle)
        angle = float(np.median(angles)) if angles else 0.0
        return _rotate_same_size(image, angle), angle