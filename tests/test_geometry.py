from pathlib import Path
import sys
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attendance.geometry import deskew


def test_deskew_preserves_image_size():
    image = cv2.imread(str(ROOT / "input" / "5.jpeg"))

    corrected, angle = deskew(image)

    assert corrected.shape == image.shape
    assert isinstance(angle, float)
    assert abs(angle) < 10