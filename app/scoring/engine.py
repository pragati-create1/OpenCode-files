"""Transparent weighted scoring engine."""
from __future__ import annotations

from app.utils.config import weights, verdict_thresholds, load_config


def combine(signals: dict) -> dict:
    w = weights()
    thr = verdict_thresholds()
    num, den = 0.0, 0.0
    contributions = {}
    for key, sig in signals.items():
        if not isinstance(sig, dict):
            continue
        score = sig.get("score")
        if score is None or sig.get("confidence", 0) <= 0:
            contributions[key] = {"used": False, "reason": "insufficient evidence"}
            continue
        wk = w.get(key, 0.1) * max(sig.get("confidence", 0.5), 0.1)
        num += score * wk
        den += wk
        contributions[key] = {"used": True, "score": round(float(score), 1),
                              "weight": round(wk, 3),
                              "supports": "authenticity" if score >= 65 else ("manipulation" if score <= 40 else "neutral/uncertain")}
    if den == 0:
        return {"authenticity_score": None, "verdict": "Uncertain", "confidence": 0.0,
                "contributions": contributions, "note": "No signal provided usable evidence."}
    final = num / den
    # verdict
    if final >= thr["real_threshold"]:
        verdict = "Likely Real"
    elif final <= thr["fake_threshold"]:
        verdict = "Likely Manipulated"
    else:
        verdict = "Uncertain"
    # confidence: distance from decision thresholds + agreement of used signals
    margin = min(abs(final - thr["real_threshold"]), abs(final - thr["fake_threshold"]))
    used = [c for c in contributions.values() if c["used"]]
    conf = min(0.95, 0.4 + margin / 60.0 + 0.1 * min(len(used), 5))
    if verdict == "Uncertain":
        conf = min(conf, 0.6)
    return {
        "authenticity_score": round(final, 1),
        "verdict": verdict,
        "confidence": round(float(conf), 2),
        "contributions": contributions,
    }
