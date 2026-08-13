#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from attendance.visualization import create_student_summary

def main():
 p=argparse.ArgumentParser();p.add_argument('student_index');p.add_argument('--db',default=str(ROOT/'data/attendance.db'));p.add_argument('--output',default=str(ROOT/'outputs/charts'));a=p.parse_args()
 paths=create_student_summary(a.db,a.student_index,a.output)
 print('Created:');[print(' -',x) for x in paths]
if __name__=='__main__':main()
