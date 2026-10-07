from app.scoring.engine import combine


def test_scoring_authentic_signals():
    signals = {
        "compression": {"name": "compression", "score": 85, "confidence": 0.5, "explanation": "ok"},
        "noise": {"name": "noise", "score": 90, "confidence": 0.5, "explanation": "ok"},
    }
    r = combine(signals)
    assert r["verdict"] == "Likely Real"
    assert 0 < r["confidence"] <= 1
    assert r["authenticity_score"] >= 68


def test_scoring_fake_signals():
    signals = {
        "face": {"name": "face", "score": 25, "confidence": 0.6, "explanation": "bad"},
        "temporal": {"name": "temporal", "score": 30, "confidence": 0.6, "explanation": "bad"},
    }
    r = combine(signals)
    assert r["verdict"] == "Likely Manipulated"


def test_scoring_no_evidence():
    r = combine({"deepfake_model": {"name": "deepfake_model", "score": None, "confidence": 0.0, "explanation": "insufficient"}})
    assert r["verdict"] == "Uncertain"
    assert r["authenticity_score"] is None
