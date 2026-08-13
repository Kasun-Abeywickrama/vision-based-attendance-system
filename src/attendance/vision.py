from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class VisionConfig:
    work_width: int = 1400
    dark_threshold: int = 175
    presence_ink_ratio: float = 0.020
    uncertain_band: float = 0.006
    save_intermediate: bool = True


class SheetDetectionError(RuntimeError):
    pass


def _cluster(values: list[int], tolerance: int = 4) -> list[int]:
    if not values:
        return []

    values = sorted(values)
    groups = [[values[0]]]

    for value in values[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])

    return [int(np.median(group)) for group in groups]


class AttendanceVision:
    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()

    def _find_table(
        self,
        image: np.ndarray,
        student_count: int
    ):
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12
        )

        y_offset = int(0.25 * h)
        x_offset = int(0.05 * w)

        search = binary[
            y_offset:int(0.55 * h),
            x_offset:int(0.90 * w)
        ]

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(100, int(0.16 * w)), 1)
        )

        horizontal = cv2.morphologyEx(
            search,
            cv2.MORPH_OPEN,
            horizontal_kernel
        )

        counts = (horizontal > 0).sum(axis=1)

        raw_y = np.where(
            counts > int(0.28 * w)
        )[0].tolist()

        y_candidates = [
            y + y_offset
            for y in _cluster(raw_y, 4)
        ]

        needed = student_count + 2
        best: tuple[float, list[int]] | None = None

        for i in range(max(0, len(y_candidates) - needed + 1)):
            seq = y_candidates[i:i + needed]

            if len(seq) != needed:
                continue

            gaps = np.diff(seq)
            median_gap = float(np.median(gaps))

            if not (18 <= median_gap <= 70):
                continue

            max_deviation = max(
                12.0,
                median_gap * 0.33
            )

            if float(
                np.max(np.abs(gaps - median_gap))
            ) > max_deviation:
                continue

            score = float(
                np.std(gaps) / (median_gap + 1e-9)
                - 0.002 * seq[0]
            )

            if best is None or score < best[0]:
                best = (score, seq)

        if best is None:
            raise SheetDetectionError(
                f"Could not identify a {student_count}-student table. "
                f"Horizontal candidates={y_candidates}"
            )

        ys = best[1]


def compute_cell_bounds(
    x_left: int,
    x_right: int,
    y_top: int,
    y_bottom: int,
    margin_x_ratio: float = 0.07,
    margin_y_ratio: float = 0.18,
) -> tuple[int, int, int, int]:
    """Shrink a raw table cell inward so grid-line ink is excluded."""
    margin_x = max(4, int((x_right - x_left) * margin_x_ratio))
    margin_y = max(3, int((y_bottom - y_top) * margin_y_ratio))

    return (
        x_left + margin_x,
        x_right - margin_x,
        y_top + margin_y,
        y_bottom - margin_y,
    )


def extract_signature_roi(
    image: np.ndarray,
    x_left: int,
    x_right: int,
    y_top: int,
    y_bottom: int,
) -> np.ndarray:
    """Crop a single student's signature cell, excluding grid-line margins."""
    xa, xb, ya, yb = compute_cell_bounds(x_left, x_right, y_top, y_bottom)
    return image[ya:yb, xa:xb]


def save_signature_roi(
    roi: np.ndarray,
    output_dir: str | Path,
    row_index: int,
    student_index: str,
) -> Path:
    """Save a cropped signature ROI as row_<n>_<student index>.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_path = output_dir / f"row_{row_index}_{student_index}.png"
    cv2.imwrite(str(roi_path), roi)
    return roi_path


def extract_signature_rois(
    image: np.ndarray,
    row_boundaries: list[int],
    x_left: int,
    x_right: int,
    student_indices: list[str],
    output_dir: str | Path,
) -> list[Path]:
    """Crop and save one signature ROI per student row.

    ``row_boundaries`` holds one more entry than ``student_indices``: student
    row ``i`` spans ``row_boundaries[i]`` to ``row_boundaries[i + 1]``.
    """
    if len(row_boundaries) != len(student_indices) + 1:
        raise ValueError("row_boundaries must contain one more entry than student_indices")

    roi_paths = []
    for row, student_index in enumerate(student_indices):
        y_top, y_bottom = row_boundaries[row], row_boundaries[row + 1]
        roi = extract_signature_roi(image, x_left, x_right, y_top, y_bottom)
        roi_paths.append(save_signature_roi(roi, output_dir, row + 1, student_index))

    return roi_paths