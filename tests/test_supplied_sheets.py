from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from attendance.vision import AttendanceVision
from attendance.xml_reader import load_students

EXPECTED={
 '1.jpeg':['PRESENT']*6,
 '2.jpeg':['PRESENT']*6,
 '3.jpeg':['PRESENT','ABSENT','ABSENT','PRESENT','PRESENT','PRESENT'],
 '4.jpeg':['PRESENT','ABSENT','PRESENT','ABSENT','PRESENT','PRESENT'],
 '5.jpeg':['PRESENT']*6,
}
def test_supplied_images(tmp_path):
 students=load_students(ROOT/'data/info.xml');det=AttendanceVision()
 for name,expected in EXPECTED.items():
  path=ROOT/'input'/name
  if not path.exists():continue
  results,_=det.process(path,students,tmp_path/name.replace('.jpeg',''))
  assert [r.status for r in results]==expected
