from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from .database import AttendanceDatabase


def create_student_summary(db_path: str | Path, index_no: str, output_dir: str | Path) -> list[Path]:
    db = AttendanceDatabase(db_path)
    rows = db.student_history(index_no)
    if not rows:
        raise ValueError(f"No attendance records found for student {index_no}")
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    present=sum(r['status']=='PRESENT' for r in rows)
    absent=sum(r['status']=='ABSENT' for r in rows)
    uncertain=len(rows)-present-absent

    p1=output_dir/f"{index_no}_summary.png"
    fig,ax=plt.subplots(figsize=(7,4))
    labels=['Present','Absent','Uncertain']; values=[present,absent,uncertain]
    ax.bar(labels,values)
    ax.set_ylabel('Sessions')
    ax.set_title(f'Attendance summary — {index_no}')
    ax.set_ylim(bottom=0)
    fig.tight_layout();fig.savefig(p1,dpi=160);plt.close(fig)

    p2=output_dir/f"{index_no}_history.png"
    fig,ax=plt.subplots(figsize=(9,4))
    y=[1 if r['status']=='PRESENT' else (0 if r['status']=='ABSENT' else .5) for r in rows]
    labels=[r['session_date'] or f"Session {i+1}" for i,r in enumerate(rows)]
    ax.plot(range(len(rows)),y,marker='o')
    ax.set_yticks([0,.5,1],['Absent','Uncertain','Present'])
    ax.set_xticks(range(len(rows)),labels,rotation=35,ha='right')
    ax.set_title(f'Attendance history — {index_no}')
    ax.grid(axis='y',alpha=.3)
    fig.tight_layout();fig.savefig(p2,dpi=160);plt.close(fig)
    return [p1,p2]
