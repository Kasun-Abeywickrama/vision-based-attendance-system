from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True)
class Student:
    index: str
    name: str
    title: str = ""

@dataclass(slots=True)
class DetectionResult:
    row: int
    student: Student
    status: str
    confidence: float
    ink_ratio: float
    color_ratio: float
    roi_path: Path | None = None

@dataclass(slots=True)
class SessionInfo:
    subject_code: str = "CS402.3"
    subject_name: str = "Computer Graphics and Visualization"
    session_date: str = ""
