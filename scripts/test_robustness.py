"""Robustness testing: transform media, re-analyze, compare scores.

Usage: python scripts/test_robustness.py data/samples/sample_image.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from app.core.pipeline import analyze_media
from app.utils.validation import detect_media_type

OUT = Path("data/processed/robustness")
OUT.mkdir(parents=True, exist_ok=True)


def transform_image(src: Path) -> list[Path]:
    img = cv2.imread(str(src))
    variants = []
    p1 = OUT / "recompressed_q50.jpg"
    cv2.imwrite(str(p1), img, [cv2.IMWRITE_JPEG_QUALITY, 50]); variants.append(p1)
    p2 = OUT / "resized_50pct.jpg"
    cv2.imwrite(str(p2), cv2.resize(img, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85]); variants.append(p2)
    p3 = OUT / "cropped.jpg"
    h, w = img.shape[:2]
    cv2.imwrite(str(p3), img[h // 8: 7 * h // 8, w // 8: 7 * w // 8]); variants.append(p3)
    p4 = OUT / "noisy.jpg"
    import numpy as np
    noisy = cv2.add(img, (np.random.default_rng(1).normal(0, 8, img.shape)).astype(np.int16).clip(-30, 30).astype(np.uint8))
    cv2.imwrite(str(p4), noisy); variants.append(p4)
    return variants


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_robustness.py <media_path>")
        sys.exit(1)
    src = Path(sys.argv[1])
    media_type = detect_media_type(src)
    if media_type != "image":
        print("Robustness script currently handles images (video/audio supported in full pipeline).")
        sys.exit(0)
    base = analyze_media(str(src), media_type, work_dir=".")
    print(f"Original: score={base['authenticity_score']} verdict={base['verdict']}")
    rows = []
    for v in transform_image(src):
        r = analyze_media(str(v), media_type, work_dir=".")
        rows.append((v.name, r["authenticity_score"], r["verdict"]))
        print(f"  {v.name}: score={r['authenticity_score']} verdict={r['verdict']}")
    changes = sum(1 for _, _, vd in rows if vd != base["verdict"])
    print(f"Verdict changes: {changes}/{len(rows)}")


if __name__ == "__main__":
    main()
