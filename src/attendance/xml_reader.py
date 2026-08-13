from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from .models import Student


def _text(node, names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        found = node.find(name)
        if found is not None and found.text:
            return found.text.strip()
    return default


def load_students(path: str | Path) -> list[Student]:
    """Load students from a permissive XML format.

    Supported student element names include ``student`` and nested structures.
    Index tags accepted: index, indexno, index_no, student_no, id.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    nodes = root.findall(".//student")
    if not nodes:
        raise ValueError(f"No <student> records found in {path}")

    students: list[Student] = []
    for node in nodes:
        index = _text(node, ("index", "indexno", "index_no", "student_no", "id"))
        name = _text(node, ("name", "student_name"))
        title = _text(node, ("title",))
        if not index:
            raise ValueError("Each <student> requires an index/indexno field")
        students.append(Student(index=index, name=name or f"Student {index}", title=title))
    return students
