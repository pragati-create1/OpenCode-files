from app.core.report import to_json, to_html


def _result():
    return {"filename": "x.png", "media_type": "image", "analyzed_at": "2026-10-06",
            "authenticity_score": 78, "verdict": "Likely Real", "confidence": 0.8,
            "signals": {"noise": {"score": 80, "confidence": 0.5, "explanation": "ok"}},
            "evidence": [{"signal": "noise", "explanation": "ok"}], "limitations": ["L"]}


def test_report_json():
    assert '"authenticity_score": 78' in to_json(_result())


def test_report_html():
    h = to_html(_result())
    assert "Forensic Report" in h and "Likely Real" in h
