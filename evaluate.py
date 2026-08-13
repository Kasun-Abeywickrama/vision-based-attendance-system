#!/usr/bin/env python3
from __future__ import annotations
import csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from attendance.vision import AttendanceVision
from attendance.xml_reader import load_students

EXPECTED={
 '1.jpeg':['PRESENT']*6,
 '2.jpeg':['PRESENT']*6,
 '3.jpeg':['PRESENT','ABSENT','ABSENT','PRESENT','PRESENT','PRESENT'],
 '4.jpeg':['PRESENT','ABSENT','PRESENT','ABSENT','PRESENT','PRESENT'],
 '5.jpeg':['PRESENT']*6,
}

def main():
 students=load_students(ROOT/'data/info.xml');det=AttendanceVision();rows=[]
 tp=tn=fp=fn=0
 for name,expected in EXPECTED.items():
  results,_=det.process(ROOT/'input'/name,students,ROOT/'outputs'/'evaluation'/Path(name).stem)
  for r,truth in zip(results,expected):
   pred=r.status
   rows.append([name,r.row,r.student.index,truth,pred,r.ink_ratio,r.confidence])
   if truth=='PRESENT' and pred=='PRESENT':tp+=1
   elif truth=='ABSENT' and pred=='ABSENT':tn+=1
   elif truth=='ABSENT' and pred=='PRESENT':fp+=1
   elif truth=='PRESENT' and pred=='ABSENT':fn+=1
 total=tp+tn+fp+fn
 accuracy=(tp+tn)/total if total else 0
 precision=tp/(tp+fp) if tp+fp else 0
 recall=tp/(tp+fn) if tp+fn else 0
 f1=2*precision*recall/(precision+recall) if precision+recall else 0
 out=ROOT/'outputs'/'evaluation';out.mkdir(parents=True,exist_ok=True)
 with (out/'evaluation.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['image','row','student_index','ground_truth','prediction','ink_ratio','confidence']);w.writerows(rows)
 report=(f"Total cells: {total}\nTP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}\n"
         f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n")
 (out/'metrics.txt').write_text(report,encoding='utf-8');print(report)
 print('Saved:',out/'evaluation.csv');print('Saved:',out/'metrics.txt')
if __name__=='__main__':main()
