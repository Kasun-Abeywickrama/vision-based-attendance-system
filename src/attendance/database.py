from __future__ import annotations
import sqlite3
from pathlib import Path
from .models import DetectionResult, SessionInfo, Student

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_no TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    title TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_code TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    session_date TEXT NOT NULL,
    image_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('PRESENT','ABSENT','UNCERTAIN')),
    confidence REAL NOT NULL,
    ink_ratio REAL NOT NULL,
    color_ratio REAL NOT NULL,
    signature_roi_path TEXT,
    UNIQUE(session_id, student_id)
);
CREATE TABLE IF NOT EXISTS signature_verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attendance_id INTEGER NOT NULL UNIQUE REFERENCES attendance(id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    verification_status TEXT NOT NULL,
    review_required INTEGER NOT NULL DEFAULT 0,
    reference_count INTEGER NOT NULL DEFAULT 0,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

class AttendanceDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_session(self, students: list[Student], results: list[DetectionResult], session: SessionInfo, image_path: str) -> int:
        with self.connect() as conn:
            for s in students:
                conn.execute(
                    "INSERT INTO students(index_no,name,title) VALUES(?,?,?) "
                    "ON CONFLICT(index_no) DO UPDATE SET name=excluded.name,title=excluded.title",
                    (s.index, s.name, s.title),
                )
            cur = conn.execute(
                "INSERT INTO sessions(subject_code,subject_name,session_date,image_path) VALUES(?,?,?,?)",
                (session.subject_code, session.subject_name, session.session_date, str(image_path)),
            )
            session_id = int(cur.lastrowid)
            for r in results:
                sid = conn.execute("SELECT id FROM students WHERE index_no=?", (r.student.index,)).fetchone()[0]
                conn.execute(
                    "INSERT INTO attendance(session_id,student_id,status,confidence,ink_ratio,color_ratio,signature_roi_path) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (session_id, sid, r.status, r.confidence, r.ink_ratio, r.color_ratio,
                     str(r.roi_path) if r.roi_path else None),
                )
            return session_id

    def student_history(self, index_no: str):
        with self.connect() as conn:
            return conn.execute(
                """SELECT s.session_date,s.subject_code,s.subject_name,a.status,a.confidence,a.ink_ratio
                   FROM attendance a
                   JOIN students st ON st.id=a.student_id
                   JOIN sessions s ON s.id=a.session_id
                   WHERE st.index_no=? ORDER BY s.session_date,s.id""",
                (index_no,),
            ).fetchall()

    def signature_history(self, index_no: str, *, present_only: bool = True, limit: int = 8) -> list[str]:
        """Return readable historical ROI paths for use as signature references."""
        status_filter = "AND a.status='PRESENT'" if present_only else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT a.signature_roi_path
                    FROM attendance a
                    JOIN students st ON st.id=a.student_id
                    JOIN sessions s ON s.id=a.session_id
                    WHERE st.index_no=?
                      AND a.signature_roi_path IS NOT NULL
                      {status_filter}
                    ORDER BY s.session_date DESC,s.id DESC
                    LIMIT ?""",
                (index_no, limit),
            ).fetchall()
        return [r["signature_roi_path"] for r in rows if r["signature_roi_path"] and Path(r["signature_roi_path"]).exists()]

    def save_signature_verification(self, session_id: int, index_no: str, result: dict) -> None:
        """Persist an optional proxy-signing assessment without altering attendance."""
        import json
        with self.connect() as conn:
            row = conn.execute(
                """SELECT a.id AS attendance_id
                   FROM attendance a
                   JOIN students st ON st.id=a.student_id
                   WHERE a.session_id=? AND st.index_no=?""",
                (session_id, index_no),
            ).fetchone()
            if row is None:
                raise ValueError(f"Attendance row not found for session {session_id}, student {index_no}")
            conn.execute(
                """INSERT INTO signature_verifications(
                       attendance_id,similarity,verification_status,review_required,reference_count,details_json
                   ) VALUES(?,?,?,?,?,?)
                   ON CONFLICT(attendance_id) DO UPDATE SET
                       similarity=excluded.similarity,
                       verification_status=excluded.verification_status,
                       review_required=excluded.review_required,
                       reference_count=excluded.reference_count,
                       details_json=excluded.details_json""",
                (
                    row["attendance_id"],
                    float(result.get("similarity", 0.0)),
                    str(result.get("status", "INCONCLUSIVE")),
                    int(bool(result.get("review_required", False))),
                    int(result.get("reference_count", 0)),
                    json.dumps(result),
                ),
            )
