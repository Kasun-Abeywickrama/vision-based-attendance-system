from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attendance.vision import (
    compute_cell_bounds,
    extract_signature_roi,
    extract_signature_rois,
)


def test_compute_cell_bounds_shrinks_inward():
    xa, xb, ya, yb = compute_cell_bounds(100, 300, 50, 150)

    assert 100 < xa < xb < 300
    assert 50 < ya < yb < 150


def test_extract_signature_roi_matches_margin_bounds():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    xa, xb, ya, yb = compute_cell_bounds(50, 350, 20, 180)

    roi = extract_signature_roi(image, 50, 350, 20, 180)

    assert roi.shape[:2] == (yb - ya, xb - xa)


def test_extract_signature_rois_writes_one_file_per_student(tmp_path):
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    row_boundaries = [40, 90, 140, 190]
    student_indices = ["10000409", "10009301", "10009302"]

    roi_paths = extract_signature_rois(
        image,
        row_boundaries,
        x_left=60,
        x_right=340,
        student_indices=student_indices,
        output_dir=tmp_path,
    )

    assert len(roi_paths) == 3

    for path, student_index in zip(roi_paths, student_indices):
        assert path.exists()
        assert student_index in path.name
