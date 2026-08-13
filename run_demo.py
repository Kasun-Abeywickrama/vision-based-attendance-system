#!/usr/bin/env python3
from __future__ import annotations
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATES={'1.jpeg':'2019-07-31','2.jpeg':'2019-08-26','3.jpeg':'2019-08-28','4.jpeg':'2019-07-05','5.jpeg':'2019-07-12'}

def run(args):
 print('\n$',' '.join(map(str,args)));subprocess.run(args,cwd=ROOT,check=True)

def main():
 db=ROOT/'data'/'attendance.db'
 if db.exists():db.unlink()
 for name,day in DATES.items():
  run([sys.executable,'sams.py',f'input/{name}','data/info.xml','--date',day])
 run([sys.executable,'infovis.py','10009301'])
 run([sys.executable,'evaluate.py'])
 print('\nDemo completed successfully.')
if __name__=='__main__':main()
