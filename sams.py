#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from datetime import date
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from attendance.database import AttendanceDatabase
from attendance.models import SessionInfo
from attendance.vision import AttendanceVision,VisionConfig
from attendance.xml_reader import load_students
from attendance.verification import assess_proxy_signature


def main():
    p=argparse.ArgumentParser(description='Student Attendance Management System')
    p.add_argument('image',help='Photographed signing sheet')
    p.add_argument('xml',help='Student information XML')
    p.add_argument('--db',default=str(ROOT/'data/attendance.db'))
    p.add_argument('--output',default=None)
    p.add_argument('--date',dest='session_date',default=date.today().isoformat())
    p.add_argument('--subject-code',default='CS402.3')
    p.add_argument('--subject-name',default='Computer Graphics and Visualization')
    p.add_argument('--threshold',type=float,default=0.020,help='Signature ink-ratio presence threshold')
    p.add_argument('--proxy-check',action='store_true',help='Optionally compare PRESENT signatures with trusted/historical references; attendance status is never changed')
    p.add_argument('--reference-dir',default=None,help='Optional directory containing per-student reference folders, e.g. references/10000409/*.png')
    p.add_argument('--min-references',type=int,default=2,help='Minimum reference signatures required for a proxy-signing decision')
    args=p.parse_args()

    students=load_students(args.xml)
    out=Path(args.output) if args.output else ROOT/'outputs'/Path(args.image).stem
    print(f'[1/7] Loaded {len(students)} students from {args.xml}')
    detector=AttendanceVision(VisionConfig(presence_ink_ratio=args.threshold))
    print('[2/7] Detecting and deskewing sheet...')
    results,meta=detector.process(args.image,students,out)
    print(f"[3/7] Student table detected (deskew {meta['deskew_angle_degrees']:.2f}°)")
    print('[4/7] Signature cells extracted')
    print('[5/7] Attendance classified')

    db=AttendanceDatabase(args.db)

    # Optional proxy-signing check. It is intentionally advisory only and does
    # not alter the existing PRESENT/ABSENT/UNCERTAIN classification. Historical
    # references are read before the new session is saved, preventing the
    # candidate image from becoming one of its own references.
    proxy_results = {}
    if args.proxy_check:
        ref_root = Path(args.reference_dir) if args.reference_dir else None
        for r in results:
            if r.status != 'PRESENT' or not r.roi_path:
                continue
            refs = []
            if ref_root is not None:
                student_dir = ref_root / r.student.index
                if student_dir.exists():
                    refs = sorted(str(x) for x in student_dir.iterdir() if x.is_file() and x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
            if not refs:
                refs = db.signature_history(r.student.index, present_only=True, limit=8)
            proxy_results[r.student.index] = assess_proxy_signature(
                r.roi_path, refs, min_references=max(1,args.min_references)
            )

    session=SessionInfo(args.subject_code,args.subject_name,args.session_date)
    sid=db.save_session(students,results,session,args.image)
    print(f'[6/7] Saved as database session #{sid}')
    for index_no, verification in proxy_results.items():
        db.save_signature_verification(sid,index_no,verification)

    csv_path=out/'attendance.csv';out.mkdir(parents=True,exist_ok=True)
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['row','index','name','status','confidence','ink_ratio','color_ratio','roi'])
        for r in results:w.writerow([r.row,r.student.index,r.student.name,r.status,f'{r.confidence:.4f}',f'{r.ink_ratio:.6f}',f'{r.color_ratio:.6f}',r.roi_path])
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

    if args.proxy_check:
        proxy_csv = out/'proxy_verification.csv'
        with proxy_csv.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['index','status','similarity','review_required','reference_count'])
            for r in results:
                v=proxy_results.get(r.student.index)
                if v:
                    w.writerow([r.student.index,v['status'],f"{v['similarity']:.4f}",v['review_required'],v['reference_count']])
                else:
                    w.writerow([r.student.index,'NOT_CHECKED','',False,0])

    print('\nRow  Index       Status      Conf.   Ink ratio  Name')
    print('-'*78)
    for r in results:
        print(f'{r.row:<4} {r.student.index:<11} {r.status:<11} {r.confidence:>6.1%}   {r.ink_ratio:>8.4f}  {r.student.name}')
    if args.proxy_check:
        print('\nProxy-signing review (advisory only):')
        for r in results:
            v=proxy_results.get(r.student.index)
            if v:
                print(f"  {r.student.index}: {v['status']:<22} similarity={v['similarity']:.1%} refs={v['reference_count']}")
        print(f'Proxy CSV: {out/"proxy_verification.csv"}')
    print(f"\n[7/7] Done. Results: {out}")
    print(f'CSV: {csv_path}')
    print(f'Database: {args.db}')

if __name__=='__main__':main()
