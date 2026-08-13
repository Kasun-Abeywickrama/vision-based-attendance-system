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


def _rotate_same_size(
    image: np.ndarray,
    angle: float
) -> np.ndarray:
    h, w = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


class AttendanceVision:
    """Classical-CV detector specialized for the static signing-sheet layout.

    It automatically deskews the photographed sheet, identifies the regular
    student table from its horizontal grid lines, locates the right-most
    signature column, then measures handwriting inside each student cell.
    """

    def __init__(
        self,
        config: VisionConfig | None = None
    ):
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
    
    def _find_table(
        self,
        image: np.ndarray,
        student_count: int
    ) -> tuple[
        list[int],
        tuple[int, int],
        np.ndarray,
        np.ndarray
    ]:
        h, w = image.shape[:2]

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

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
            (
                max(100, int(0.16 * w)),
                1
            )
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

        best: tuple[
            float,
            list[int]
        ] | None = None

        for i in range(
            max(
                0,
                len(y_candidates) - needed + 1
            )
        ):
            seq = y_candidates[
                i:i + needed
            ]

            if len(seq) != needed:
                continue

            gaps = np.diff(seq)

            median_gap = float(
                np.median(gaps)
            )

            if not (
                18 <= median_gap <= 70
            ):
                continue

            max_deviation = max(
                12.0,
                median_gap * 0.33
            )

            if float(
                np.max(
                    np.abs(
                        gaps - median_gap
                    )
                )
            ) > max_deviation:
                continue

            score = float(
                np.std(gaps)
                / (median_gap + 1e-9)
                - 0.002 * seq[0]
            )

            if (
                best is None
                or score < best[0]
            ):
                best = (
                    score,
                    seq
                )

        if best is None:
            raise SheetDetectionError(
                f"Could not identify a "
                f"{student_count}-student table. "
                f"Horizontal candidates="
                f"{y_candidates}"
            )

        ys = best[1]

        y0, y1 = ys[0], ys[-1]

        table_strip = binary[
            y0:y1 + 1,
            x_offset:int(0.90 * w)
        ]

        vertical_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (
                    1,
                    max(
                        15,
                        int(
                            (y1 - y0) * 0.55
                        )
                    )
                )
            )
        )

        vertical = cv2.morphologyEx(
            table_strip,
            cv2.MORPH_OPEN,
            vertical_kernel
        )

        x_counts = (
            vertical > 0
        ).sum(axis=0)

        raw_x = np.where(
            x_counts
            > int((y1 - y0) * 0.42)
        )[0].tolist()

        x_candidates = [
            x + x_offset
            for x in _cluster(raw_x, 4)
        ]

        if len(x_candidates) < 2:
            raise SheetDetectionError(
                "Could not locate signature column. "
                f"Vertical candidates="
                f"{x_candidates}"
            )

        # Last two table boundaries define
        # the signature column.
        x_left, x_right = sorted(
            x_candidates
        )[-2:]

        if x_right - x_left < 50:
            raise SheetDetectionError(
                f"Signature column too narrow: "
                f"{x_left}..{x_right}"
            )

        return (
            ys,
            (x_left, x_right),
            gray,
            binary
        )

    # ============================================================
    # MEMBER 6
    # Signature Feature Extraction
    # ============================================================

    def _extract_signature_features(
        self,
        roi: np.ndarray
    ) -> tuple[float, float]:
        """Extract numerical features from a signature ROI.

        Returns:
            ink_ratio:
                Ratio of dark/ink pixels inside the ROI.

            color_ratio:
                Ratio of pixels satisfying the HSV-based
                colour/ink condition.
        """

        # Convert the signature ROI to grayscale.
        roi_gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )
        

        # Calculate the proportion of dark pixels.
        # This is the main feature used for attendance
        # classification.
        ink_ratio = float(
            (
                roi_gray
                < self.config.dark_threshold
            ).mean()
        )

        # Convert ROI to HSV for an additional
        # colour-based signature measurement.
        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )

        color_ratio = float(
            (
                (hsv[:, :, 1] > 45)
                & (hsv[:, :, 2] < 245)
            ).mean()
        )

        return ink_ratio, color_ratio

    # ============================================================
    # MEMBER 6
    # Attendance Classification
    # ============================================================

    def _classify_attendance(
        self,
        ink_ratio: float
    ) -> tuple[str, float]:
        """Classify attendance using the signature ink ratio.

        Classification:
            PRESENT:
                ink ratio is above the presence threshold
                plus the uncertainty band.

            ABSENT:
                ink ratio is below the presence threshold
                minus the uncertainty band.

            UNCERTAIN:
                ink ratio falls inside the uncertainty band.

        Returns:
            status:
                PRESENT, ABSENT, or UNCERTAIN.

            confidence:
                Confidence value between the configured
                minimum calculation and 0.99.
        """

        threshold = (
            self.config.presence_ink_ratio
        )

        band = (
            self.config.uncertain_band
        )

        # Attendance decision.
        if ink_ratio >= threshold + band:
            status = "PRESENT"

        elif ink_ratio <= max(
            0.0,
            threshold - band
        ):
            status = "ABSENT"

        else:
            status = "UNCERTAIN"

        # Calculate how far the observed
        # ink ratio is from the decision threshold.
        distance = (
            abs(ink_ratio - threshold)
            / (
                band
                if band > 0
                else threshold
            )
        )

        # Convert the distance into a confidence score.
        confidence = float(
            min(
                0.99,
                0.55 + 0.11 * distance
            )
        )

        return status, confidence

    def process(
        self,
        image_path: str | Path,
        students: list[Student],
        output_dir: str | Path
    ) -> tuple[
        list[DetectionResult],
        dict
    ]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        raw = cv2.imread(
            str(image_path)
        )

        if raw is None:
            raise ValueError(
                f"Unable to read image: "
                f"{image_path}"
            )

        h0, w0 = raw.shape[:2]

        scale = (
            self.config.work_width
            / w0
        )

        work = cv2.resize(
            raw,
            (
                self.config.work_width,
                int(h0 * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

        deskewed, angle = self._deskew(
            work
        )

        (
            ys,
            (x_left, x_right),
            gray,
            binary
        ) = self._find_table(
            deskewed,
            len(students)
        )

        overlay = deskewed.copy()
        table_overlay = deskewed.copy()

        for y in ys:
            cv2.line(
                table_overlay,
                (0, y),
                (
                    table_overlay.shape[1] - 1,
                    y
                ),
                (0, 0, 255),
                2
            )

        cv2.line(
            table_overlay,
            (x_left, ys[0]),
            (x_left, ys[-1]),
            (255, 0, 0),
            3
        )

        cv2.line(
            table_overlay,
            (x_right, ys[0]),
            (x_right, ys[-1]),
            (255, 0, 0),
            3
        )

        results: list[DetectionResult] = []

        roi_dir = (
            output_dir
            / "signature_rois"
        )

        roi_dir.mkdir(
            exist_ok=True
        )

        for row, student in enumerate(
            students
        ):
            # ----------------------------------------------------
            # ROI extraction
            # ----------------------------------------------------
            ya, yb = (
                ys[row + 1],
                ys[row + 2]
            )

            # Exclude grid borders while retaining
            # enough writing near the edges.
            mx = max(
                4,
                int(
                    (x_right - x_left)
                    * 0.07
                )
            )

            my = max(
                3,
                int(
                    (yb - ya)
                    * 0.18
                )
            )

            xa, xb = (
                x_left + mx,
                x_right - mx
            )

            yc, yd = (
                ya + my,
                yb - my
            )

            roi = deskewed[
                yc:yd,
                xa:xb
            ]

            # ----------------------------------------------------
            # MEMBER 6:
            # Signature feature extraction
            # ----------------------------------------------------
            ink_ratio, color_ratio = (
                self._extract_signature_features(
                    roi
                )
            )

            # ----------------------------------------------------
            # MEMBER 6:
            # Attendance classification
            # ----------------------------------------------------
            
        return ys,(x_left,x_right),gray,binary

    def process(self, image_path: str | Path, students: list[Student], output_dir: str | Path) -> tuple[list[DetectionResult], dict]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        raw = cv2.imread(str(image_path))
        if raw is None:
            raise ValueError(f"Unable to read image: {image_path}")

        h0,w0 = raw.shape[:2]
        scale = self.config.work_width / w0
        work = cv2.resize(raw,(self.config.work_width,int(h0*scale)),interpolation=cv2.INTER_AREA)
        deskewed, angle = self._deskew(work)
        deskewed, angle = self._deskew(work)

        paths = {}
        if self.config.save_intermediate:
                    stages = {
                        "01_original.jpg": work,
                        "02_deskewed.jpg": deskewed,
                        "03_grayscale.jpg": gray,
                        "04_binary.jpg": binary,
                        "05_table_detection.jpg": table_overlay,
                        "06_results.jpg": overlay,
                    }
                    for name,img in stages.items():
                        p=output_dir/name;cv2.imwrite(str(p),img);paths[name]=str(p)

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
