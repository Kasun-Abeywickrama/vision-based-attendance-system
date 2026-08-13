#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from attendance.verification import compare_signatures,assess_proxy_signature

def main():
 p=argparse.ArgumentParser(description='Exploratory signature verification');p.add_argument('candidate');p.add_argument('references',nargs='+');p.add_argument('--proxy-assessment',action='store_true',help='Use conservative multi-reference proxy-signing assessment');p.add_argument('--min-references',type=int,default=2);a=p.parse_args();result=assess_proxy_signature(a.candidate,a.references,min_references=max(1,a.min_references)) if a.proxy_assessment else compare_signatures(a.candidate,a.references);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
