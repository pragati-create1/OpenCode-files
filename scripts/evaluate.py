"""Evaluation framework.

Layout expected:
  data/eval/<split>/<real|fake>/<image|video|audio>/<files>
If the folders are empty or missing, prints 'Evaluation not yet performed.'
Runs the real pipeline and computes accuracy/precision/recall/F1/ROC-AUC/FPR/FNR.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.pipeline import analyze_media
from app.utils.validation import detect_media_type

EVAL = Path("data/eval")
OUT = Path("data/results/evaluation.json")


def main():
    samples = []
    for label_dir in ("real", "fake"):
        for p in EVAL.rglob("*"):
            if p.is_file() and label_dir in p.parts:
                try:
                    samples.append((p, 1 if label_dir == "real" else 0))
                except Exception:
                    pass
    if not samples:
        print("Evaluation not yet performed: no labelled data under data/eval/.")
        return
    y_true, y_score = [], []
    for p, label_real in samples:
        try:
            r = analyze_media(str(p), detect_media_type(p), work_dir=".")
            s = r.get("authenticity_score")
            if s is None:
                continue
            y_true.append(label_real)
            y_score.append(s / 100.0)
        except Exception as e:
            print(f"skip {p}: {e}")
    if not y_true:
        print("Evaluation not yet performed: no analyzable samples.")
        return
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else None
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    report = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
        "confusion_matrix": cm,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
