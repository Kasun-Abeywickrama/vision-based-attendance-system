from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.attendance.vision import VisionConfig, AttendanceVision


@pytest.fixture
def dummy_image():
    # Create a dummy image
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), -1)
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
    cv2.imwrite(path, img)
    yield Path(path)
    if os.path.exists(path):
        os.remove(path)


def test_vision_config():
    config = VisionConfig(work_width=1000)
    assert config.work_width == 1000


def test_attendance_vision_process(dummy_image):
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        config = VisionConfig(work_width=400)
        vision = AttendanceVision(config)
        
        paths, metadata = vision.process(dummy_image, [], output_dir)
        
        assert "01_original.jpg" in paths
        assert "03_grayscale.jpg" in paths
        
        original_path = Path(paths["01_original.jpg"])
        grayscale_path = Path(paths["03_grayscale.jpg"])
        
        assert original_path.exists()
        assert grayscale_path.exists()
        
        # Load and verify dimensions
        orig_img = cv2.imread(str(original_path))
        gray_img = cv2.imread(str(grayscale_path), cv2.IMREAD_GRAYSCALE)
        
        # Original width should be scaled to 400
        assert orig_img.shape[1] == 400
        # Grayscale should have the same dimensions
        assert gray_img.shape == orig_img.shape[:2]
        
        assert metadata["original_dimensions"] == (300, 600)
