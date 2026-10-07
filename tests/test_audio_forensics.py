from app.detectors.audio_forensics import analyze_audio


def test_audio_analysis(sample_wav):
    r = analyze_audio(sample_wav)
    assert "audio" in r["signals"]
    a = r["signals"]["audio"]
    assert 0 <= a["score"] <= 100
    assert a["explanation"]
