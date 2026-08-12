from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attendance.preprocessing import load_image, resize_to_width, to_grayscale


def test_preprocessing_pipeline():
    image = load_image(ROOT / "input" / "5.jpeg")

    resized = resize_to_width(image, 1400)
    assert resized.shape[1] == 1400

    gray = to_grayscale(resized)
    assert len(gray.shape) == 2
    assert gray.shape[:2] == resized.shape[:2]