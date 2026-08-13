from __future__ import annotations
from pathlib import Path
from statistics import median
import cv2
import numpy as np


def _normalize(path: str | Path) -> np.ndarray:
    """Normalize a signature crop to a centered binary canvas.

    This keeps the original coursework behaviour (Otsu + tight crop + resize)
    while making the normalized image reusable by additional comparison methods.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read signature image: {path}")
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(bw)
    if pts is None:
        return cv2.resize(bw, (320, 120))
    x, y, w, h = cv2.boundingRect(pts)
    crop = bw[y:y + h, x:x + w]
    canvas = np.zeros((120, 320), dtype=np.uint8)
    scale = min(300 / max(1, w), 100 / max(1, h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    yy = (120 - nh) // 2
    xx = (320 - nw) // 2
    canvas[yy:yy + nh, xx:xx + nw] = resized
    return canvas


def _orb_score(a: np.ndarray, b: np.ndarray) -> float:
    orb = cv2.ORB_create(nfeatures=500)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None or not k1 or not k2:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    good = [m for m in matches if m.distance < 60]
    return float(min(1.0, len(good) / max(1, min(len(k1), len(k2)))))


def _projection_score(a: np.ndarray, b: np.ndarray) -> float:
    """Compare coarse horizontal/vertical ink distributions (0..1)."""
    aa = (a > 0).astype(np.float32)
    bb = (b > 0).astype(np.float32)
    pa = np.concatenate((aa.mean(axis=0), aa.mean(axis=1)))
    pb = np.concatenate((bb.mean(axis=0), bb.mean(axis=1)))
    denom = float(np.linalg.norm(pa) * np.linalg.norm(pb))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(pa, pb) / denom, 0.0, 1.0))


def _hu_score(a: np.ndarray, b: np.ndarray) -> float:
    """Shape similarity derived from Hu moments, mapped conservatively to 0..1."""
    ma = cv2.moments((a > 0).astype(np.uint8))
    mb = cv2.moments((b > 0).astype(np.uint8))
    ha = cv2.HuMoments(ma).flatten()
    hb = cv2.HuMoments(mb).flatten()

    def signed_log(v: np.ndarray) -> np.ndarray:
        return -np.sign(v) * np.log10(np.abs(v) + 1e-12)

    distance = float(np.linalg.norm(signed_log(ha) - signed_log(hb)))
    return float(np.exp(-0.35 * distance))


def _ensemble_pair_score(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Return explainable component scores plus an ensemble similarity score."""
    orb = _orb_score(candidate, reference)
    projection = _projection_score(candidate, reference)
    hu = _hu_score(candidate, reference)
    # ORB can be sparse on handwriting, so global shape features carry more weight.
    ensemble = float(np.clip(0.30 * orb + 0.45 * projection + 0.25 * hu, 0.0, 1.0))
    return {"orb": orb, "projection": projection, "hu": hu, "ensemble": ensemble}


def compare_signatures(candidate: str | Path, references: list[str | Path]) -> dict:
    """Exploratory ORB matching retained for backwards compatibility.

    Existing callers receive the same keys and decision thresholds as before.
    The proxy-signing feature below uses a separate, more conservative ensemble.
    """
    cand = _normalize(candidate)
    scores: list[float] = []
    for ref in references:
        img = _normalize(ref)
        scores.append(_orb_score(cand, img))
    score = float(max(scores) if scores else 0.0)

    if score >= 0.35:
        result = "LIKELY_MATCH"
    elif score <= 0.08:
        result = "POSSIBLE_MISMATCH"
    else:
        result = "INCONCLUSIVE"
    return {"similarity": score, "result": result, "reference_scores": scores}


def assess_proxy_signature(
    candidate: str | Path,
    references: list[str | Path],
    *,
    min_references: int = 2,
    verified_threshold: float = 0.62,
    suspicious_threshold: float = 0.30,
) -> dict:
    """Assess whether a new signature is consistent with trusted historical samples.

    This is deliberately a *review flag*, not an identity decision. Genuine
    signatures vary naturally, so low similarity never changes attendance status.
    At least ``min_references`` readable samples are required for a decisive result.
    """
    candidate_img = _normalize(candidate)
    usable_refs: list[str] = []
    pair_scores: list[dict] = []

    for ref in references:
        try:
            ref_img = _normalize(ref)
        except ValueError:
            continue
        usable_refs.append(str(ref))
        pair_scores.append(_ensemble_pair_score(candidate_img, ref_img))

    ensemble_scores = [p["ensemble"] for p in pair_scores]
    score = float(median(ensemble_scores)) if ensemble_scores else 0.0

    if len(pair_scores) < min_references:
        status = "INSUFFICIENT_REFERENCE"
        review_required = False
    elif score >= verified_threshold:
        status = "CONSISTENT"
        review_required = False
    elif score < suspicious_threshold:
        status = "SUSPICIOUS"
        review_required = True
    else:
        status = "NEEDS_REVIEW"
        review_required = True

    return {
        "similarity": score,
        "status": status,
        "review_required": review_required,
        "reference_count": len(pair_scores),
        "references": usable_refs,
        "reference_scores": pair_scores,
        "thresholds": {
            "verified": verified_threshold,
            "suspicious": suspicious_threshold,
            "min_references": min_references,
        },
    }
