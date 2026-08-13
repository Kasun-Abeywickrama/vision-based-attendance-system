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


def _rotate_same_size(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

class AttendanceVision:
    """Classical-CV detector specialized for the static signing-sheet layout.

    It automatically deskews the photographed sheet, identifies the regular
    student table from its horizontal grid lines, locates the right-most
    signature column, then measures handwriting inside each student cell.
    """
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

    def _find_table(self, image: np.ndarray, student_count: int) -> tuple[list[int], tuple[int,int], np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,12)

        y_offset, x_offset = int(0.25*h), int(0.05*w)
        search = binary[y_offset:int(0.55*h), x_offset:int(0.90*w)]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(100,int(0.16*w)),1))
        horizontal = cv2.morphologyEx(search, cv2.MORPH_OPEN, horizontal_kernel)
        counts = (horizontal > 0).sum(axis=1)
        raw_y = np.where(counts > int(0.28*w))[0].tolist()
        y_candidates = [y+y_offset for y in _cluster(raw_y,4)]

        needed = student_count + 2  # table top + header bottom + student rows
        best: tuple[float,list[int]] | None = None
        for i in range(max(0,len(y_candidates)-needed+1)):
            seq = y_candidates[i:i+needed]
            if len(seq) != needed:
                continue
            gaps = np.diff(seq)
            median_gap = float(np.median(gaps))
            if not (18 <= median_gap <= 70):
                continue
            max_deviation = max(12.0, median_gap*0.33)
            if float(np.max(np.abs(gaps-median_gap))) > max_deviation:
                continue
            # Regular spacing is primary; lower sequence is favored because the
            # student table follows the metadata table in this static form.
            score = float(np.std(gaps)/(median_gap+1e-9) - 0.002*seq[0])
            if best is None or score < best[0]:
                best = (score,seq)

        if best is None:
            raise SheetDetectionError(
                f"Could not identify a {student_count}-student table. Horizontal candidates={y_candidates}"
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
        ys,(x_left,x_right),gray,binary = self._find_table(deskewed,len(students))

        overlay = deskewed.copy()
        table_overlay = deskewed.copy()
        for y in ys:
            cv2.line(table_overlay,(0,y),(table_overlay.shape[1]-1,y),(0,0,255),2)
        cv2.line(table_overlay,(x_left,ys[0]),(x_left,ys[-1]),(255,0,0),3)
        cv2.line(table_overlay,(x_right,ys[0]),(x_right,ys[-1]),(255,0,0),3)

        results: list[DetectionResult] = []
        roi_dir = output_dir / "signature_rois"
        roi_dir.mkdir(exist_ok=True)
        for row,student in enumerate(students):
            ya,yb = ys[row+1],ys[row+2]
            # Exclude grid borders while retaining enough writing near edges.
            mx = max(4,int((x_right-x_left)*0.07))
            my = max(3,int((yb-ya)*0.18))
            xa,xb = x_left+mx,x_right-mx
            yc,yd = ya+my,yb-my
            roi = deskewed[yc:yd,xa:xb]
            roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            ink_ratio = float((roi_gray < self.config.dark_threshold).mean())
            hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
            color_ratio = float(((hsv[:,:,1]>45) & (hsv[:,:,2]<245)).mean())

            threshold = self.config.presence_ink_ratio
            band = self.config.uncertain_band
            if ink_ratio >= threshold + band:
                status = "PRESENT"
            elif ink_ratio <= max(0.0,threshold-band):
                status = "ABSENT"
            else:
                status = "UNCERTAIN"
            distance = abs(ink_ratio-threshold)/(band if band>0 else threshold)
            confidence = float(min(0.99,0.55+0.11*distance))

            roi_path = roi_dir / f"row_{row+1}_{student.index}.png"
            cv2.imwrite(str(roi_path),roi)
            results.append(DetectionResult(row+1,student,status,confidence,ink_ratio,color_ratio,roi_path))

            box_color = (0,180,0) if status=="PRESENT" else ((0,0,220) if status=="ABSENT" else (0,165,255))
            cv2.rectangle(overlay,(x_left,ya),(x_right,yb),box_color,3)
            cv2.putText(overlay,f"{student.index}: {status} {confidence:.0%}",(x_right+10,(ya+yb)//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,box_color,1,cv2.LINE_AA)

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

        meta = {
            "deskew_angle_degrees": angle,
            "horizontal_boundaries": ys,
            "signature_column": [x_left,x_right],
            "intermediate_files": paths,
        }
        return results,meta
