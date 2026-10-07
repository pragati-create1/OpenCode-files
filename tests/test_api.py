from fastapi.testclient import TestClient
from app.backend.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").status_code == 200


def test_analyze_rejects_bad_extension(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"MZ1234")
    with open(p, "rb") as f:
        r = client.post("/analyze", files={"file": ("x.exe", f)})
    assert r.status_code == 400


def test_analyze_image(sample_image):
    with open(sample_image, "rb") as f:
        r = client.post("/analyze", files={"file": ("img.png", f)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "authenticity_score" in body
    assert body["verdict"] in ("Likely Real", "Uncertain", "Likely Manipulated")
