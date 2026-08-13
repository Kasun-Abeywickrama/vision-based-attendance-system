from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from attendance.xml_reader import load_students

def test_sample_xml():
 s=load_students(ROOT/'data/info.xml');assert len(s)==6;assert s[0].index=='10000409'
