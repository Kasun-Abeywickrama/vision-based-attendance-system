from __future__ import annotations

from dataclasses import dataclass

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
        
        y0,y1 = ys[0],ys[-1]

        table_strip = binary[y0:y1+1, x_offset:int(0.90*w)]
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(15,int((y1-y0)*0.55))))
        
        vertical = cv2.morphologyEx(table_strip,cv2.MORPH_OPEN,vertical_kernel)
        
        x_counts = (vertical>0).sum(axis=0)
        
        raw_x = np.where(x_counts > int((y1-y0)*0.42))[0].tolist()
        
        x_candidates = [x+x_offset for x in _cluster(raw_x,4)]
        
        if len(x_candidates) < 2:
            raise SheetDetectionError(f"Could not locate signature column. Vertical candidates={x_candidates}")

        # Last two table boundaries define the signature column.
        x_left,x_right = sorted(x_candidates)[-2:]
        
        if x_right-x_left < 50:
            raise SheetDetectionError(f"Signature column too narrow: {x_left}..{x_right}")
            
        return ys,(x_left,x_right),gray,binary
