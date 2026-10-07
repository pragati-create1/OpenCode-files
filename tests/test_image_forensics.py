from app.detectors.image_forensics import analyze_image, detect_faces


def test_image_analysis_runs(sample_image):
    r = analyze_image(sample_image)
    for k in ("metadata", "compression", "noise", "image_stats", "face"):
        assert k in r["signals"]
        assert r["signals"][k]["explanation"]
        assert 0 <= (r["signals"][k]["score"] or 0) <= 100


def test_face_detection(sample_image):
    import cv2, numpy as np
    from app.utils.media import load_image
    rgb = load_image(sample_image)
    faces = detect_faces(rgb)
    assert isinstance(faces, list)
