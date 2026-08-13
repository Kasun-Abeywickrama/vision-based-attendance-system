from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from attendance.verification import assess_proxy_signature, compare_signatures


def _signature(path: Path, variant: int = 0):
    img=np.full((100,300,3),255,dtype=np.uint8)
    pts=np.array([[30,62],[65,42+variant],[90,66],[125,38],[155,62],[195,44],[240,58]],np.int32)
    cv2.polylines(img,[pts],False,(15,15,15),2,cv2.LINE_AA)
    cv2.line(img,(45,72),(225,72+variant),(15,15,15),2,cv2.LINE_AA)
    cv2.imwrite(str(path),img)


def test_legacy_compare_signature_api_is_preserved(tmp_path):
    a=tmp_path/'a.png';b=tmp_path/'b.png'
    _signature(a,0);_signature(b,1)
    result=compare_signatures(a,[b])
    assert set(result)=={'similarity','result','reference_scores'}
    assert result['result'] in {'LIKELY_MATCH','POSSIBLE_MISMATCH','INCONCLUSIVE'}


def test_proxy_assessment_requires_multiple_references(tmp_path):
    candidate=tmp_path/'candidate.png';ref=tmp_path/'ref.png'
    _signature(candidate,0);_signature(ref,1)
    result=assess_proxy_signature(candidate,[ref],min_references=2)
    assert result['status']=='INSUFFICIENT_REFERENCE'
    assert result['review_required'] is False
    assert result['reference_count']==1


def test_proxy_assessment_accepts_consistent_reference_set(tmp_path):
    candidate=tmp_path/'candidate.png';r1=tmp_path/'r1.png';r2=tmp_path/'r2.png';r3=tmp_path/'r3.png'
    _signature(candidate,0);_signature(r1,0);_signature(r2,1);_signature(r3,-1)
    result=assess_proxy_signature(candidate,[r1,r2,r3],min_references=2)
    assert result['reference_count']==3
    assert result['similarity']>0.5
    assert result['status'] in {'CONSISTENT','NEEDS_REVIEW'}
