"""Image forensic signals.

Every signal returns:
  name, value (measured), score (0-100, higher = more authentic-looking),
  confidence (0-1), explanation (derived only from measured values).
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

_CASCADE = None


def _face_cascade():
    global _CASCADE
    if _CASCADE is None:
        p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(p)
    return _CASCADE


def _gray_small(rgb, max_side=640):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    s = min(max_side / max(h, w), 1.0)
    return cv2.resize(g, (int(w * s), int(h * s))) if s < 1.0 else g


# ---------------- A. Metadata ----------------
def metadata_signal(path: str) -> dict:
    img = Image.open(path)
    exif = {}
    try:
        raw = img.getexif()
        for k, v in raw.items():
            tag = ExifTags.TAGS.get(k, str(k))
            exif[tag] = str(v)[:120]
    except Exception:
        pass
    software = exif.get("Software", "")
    edited_hint = any(t in software.lower() for t in ("photoshop", "gimp", "facetune", "canva"))
    dims = f"{img.width}x{img.height}"
    explanation = (
        f"Format {img.format}, size {dims}, "
        + ("EXIF fields present." if exif else "no readable EXIF fields.")
    )
    score = 80.0
    if edited_hint:
        score = 45.0
        explanation += f" Software tag '{software}' indicates editing software."
    return {
        "name": "metadata", "value": {"software": software, "format": img.format,
                                       "size": dims, "exif_fields": len(exif)},
        "score": score, "confidence": 0.6 if exif else 0.25,
        "explanation": explanation,
        "suspicious_region": None,
    }


# ---------------- B. JPEG / compression ----------------
def jpeg_signal(path: str) -> dict:
    img = Image.open(path)
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    q = getattr(img, "quantization", None)
    is_jpeg = (img.format or "").upper() in ("JPEG", "JPG")
    # blockiness: energy of horizontal/vertical differences at 8px boundaries
    if gray.shape[1] > 16 and gray.shape[0] > 16:
        b_h = np.abs(np.diff(gray[:, 7::8], axis=1)).mean() if gray.shape[1] > 16 else 0
        b_v = np.abs(np.diff(gray[7::8, :], axis=0)).mean() if gray.shape[0] > 16 else 0
        blockiness = float((b_h + b_v) / 2)
    else:
        blockiness = 0.0
    if not is_jpeg:
        return {"name": "compression", "value": {"format": img.format, "blockiness": round(blockiness, 2)},
                "score": 85.0, "confidence": 0.3,
                "explanation": f"Non-JPEG format ({img.format}); block artifact analysis not applicable.",
                "suspicious_region": None}
    # Heuristic: heavy blockiness suggests low-quality re-encode (not proof of fake).
    score = float(np.clip(100 - (blockiness - 3.0) * 6.0, 30, 95))
    return {"name": "compression", "value": {"format": "JPEG", "blockiness": round(blockiness, 2),
                                              "quantization_tables": bool(q)},
            "score": score, "confidence": 0.45,
            "explanation": f"8x8 block boundary discontinuity measured at {blockiness:.1f} "
                           f"(higher = stronger JPEG blocking). This reflects compression level, not authenticity by itself.",
            "suspicious_region": None}


# ---------------- C. Noise ----------------
def noise_signal(rgb) -> dict:
    gray = _gray_small(rgb).astype(np.float64)
    # high-pass residual as noise estimate
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    resid = gray - blur
    regions = {}
    gh, gw = np.array_split(resid, 4, axis=0), None
    # simpler: 4x4 grid of regional noise stds
    cells = []
    h, w = resid.shape
    for i in range(4):
        for j in range(4):
            cell = resid[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            cells.append(float(cell.std()))
    cells = np.array(cells)
    cv = float(cells.std() / (cells.mean() + 1e-6))
    # More uniform noise across regions => more consistent => more authentic-looking.
    score = float(np.clip(100 - cv * 120, 25, 95))
    worst = int(cells.argmax())
    region = None
    if cv > 0.45:
        r, c = divmod(worst, 4)
        region = {"grid": "4x4", "cell": [int(r), int(c)], "x": c * w // 4, "y": r * h // 4,
                  "w": w // 4, "h": h // 4}
    return {"name": "noise", "value": {"regional_noise_cv": round(cv, 3),
                                        "mean_noise_std": round(float(cells.mean()), 2)},
            "score": score, "confidence": 0.5,
            "explanation": f"Noise residual varies across regions (CV={cv:.2f}); closer to uniform is consistent with a single sensor.",
            "suspicious_region": region}


# ---------------- D. Statistics ----------------
def stats_signal(rgb) -> dict:
    gray = _gray_small(rgb)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    freq = np.fft.fftshift(np.abs(np.fft.fft2(gray.astype(np.float32))))
    h, w = freq.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2).astype(int)
    radial = np.bincount(r.ravel(), freq.ravel()) / (np.bincount(r.ravel()) + 1e-6)
    mid = len(radial) // 2
    high_ratio = float(radial[mid:].mean() / (radial[:mid].mean() + 1e-6)) if mid > 0 else 0
    score = 85.0
    notes = []
    if sharpness < 15:
        score -= 25; notes.append("very low sharpness")
    if edge_density < 0.01:
        score -= 15; notes.append("very few edges")
    if high_ratio < 0.02:
        score -= 10; notes.append("low high-frequency content")
    score = float(np.clip(score, 25, 95))
    explanation = (
        f"Sharpness {sharpness:.1f}, edge density {edge_density:.3f}, HF energy ratio {high_ratio:.3f}. "
        + (f"Notable: {', '.join(notes)}." if notes else "Within typical natural-image ranges.")
    )
    return {"name": "image_stats", "value": {"sharpness": round(sharpness, 1),
                                             "edge_density": round(edge_density, 3),
                                             "hf_ratio": round(high_ratio, 3)},
            "score": score, "confidence": 0.4, "explanation": explanation,
            "suspicious_region": None}


# ---------------- E. Face ----------------
def detect_faces(rgb) -> list:
    gray = _gray_small(rgb)
    faces = _face_cascade().detectMultiScale(gray, 1.3, 5, minSize=(40, 40))
    scale = gray.shape[1] / rgb.shape[1] if rgb.shape[1] else 1
    out = []
    for (x, y, w, h) in faces:
        out.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
    return out


def face_signal(rgb) -> dict:
    faces = detect_faces(rgb)
    if not faces:
        return {"name": "face", "value": {"faces": 0}, "score": 75.0, "confidence": 0.2,
                "explanation": "No face detected; face-specific checks not applicable.",
                "suspicious_region": None}
    textures, boundary_scores = [], []
    for (x, y, w, h) in faces:
        face = rgb[y:y + h, x:x + w]
        if face.size == 0:
            continue
        g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        textures.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        ring_in = cv2.erode(g, np.ones((7, 7), np.uint8))
        boundary = float(np.abs(g.astype(float) - cv2.GaussianBlur(g, (9, 9), 0).astype(float)).mean())
        boundary_scores.append(boundary)
    texture = float(np.mean(textures)) if textures else 0.0
    boundary = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    score = 85.0
    if texture < 5:
        score -= 25
    if boundary < 4:
        score -= 10  # overly smooth face can indicate synthesis/blending
    return {"name": "face", "value": {"faces": len(faces), "face_texture": round(texture, 1),
                                       "boundary_gradient": round(boundary, 2)},
            "score": float(np.clip(score, 25, 95)), "confidence": 0.5,
            "explanation": f"{len(faces)} face(s) found; mean face texture sharpness {texture:.1f}, "
                           f"face-boundary gradient {boundary:.1f}.",
            "suspicious_region": {"x": faces[0][0], "y": faces[0][1], "w": faces[0][2], "h": faces[0][3]}}


def analyze_image(path: str, max_side: int = 1024) -> dict:
    from app.utils.media import load_image
    rgb = load_image(path, max_side=max_side)
    signals = {
        "metadata": metadata_signal(path),
        "compression": jpeg_signal(path),
        "noise": noise_signal(rgb),
        "image_stats": stats_signal(rgb),
        "face": face_signal(rgb),
    }
    return {"signals": signals, "size": [int(rgb.shape[1]), int(rgb.shape[0])]}
