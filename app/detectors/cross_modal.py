"""Cross-modal: lip/mouth activity vs speech energy timing.

Mismatch is treated as ONE piece of evidence, not proof of manipulation
(dubbing / editing / sync errors are legitimate causes).
"""
from __future__ import annotations

import numpy as np
import cv2

from app.detectors.image_forensics import detect_faces, _gray_small


def _mouth_activity(frames_rgb) -> np.ndarray:
    vals = []
    prev = None
    for rgb in frames_rgb:
        faces = detect_faces(rgb)
        if not faces:
            vals.append(0.0)
            continue
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        lower = rgb[y + int(h * 0.55):y + h, x:x + w]
        if lower.size == 0:
            vals.append(0.0); continue
        g = cv2.cvtColor(lower, cv2.COLOR_RGB2GRAY).astype(float)
        if prev is None or prev.shape != g.shape:
            vals.append(0.0)
        else:
            vals.append(float(np.abs(g - prev).mean()))
        prev = g
    v = np.array(vals, dtype=float)
    return v / (v.max() + 1e-6)


def cross_modal_signal(frames_rgb, frame_times, audio_speech_env) -> dict:
    mouth = _mouth_activity(frames_rgb)
    if audio_speech_env is None or len(audio_speech_env["env"]) < 4:
        return {"name": "cross_modal", "value": {}, "score": 75.0, "confidence": 0.15,
                "explanation": "No usable audio envelope; cross-modal consistency could not be assessed."}
    env_t = audio_speech_env["t"]
    env = audio_speech_env["env"]
    mouth_t = np.array(frame_times)
    # resample envelope at frame times
    env_at = np.interp(mouth_t, env_t, env)
    if env_at.std() < 1e-4 or mouth.std() < 1e-4:
        return {"name": "cross_modal", "value": {}, "score": 75.0, "confidence": 0.2,
                "explanation": "Insufficient variation in mouth movement or speech energy to compare."}
    corr = float(np.corrcoef(mouth, env_at)[0, 1])
    # map correlation to a soft score: mismatch => lower authenticity score
    score = float(np.clip(75 + corr * 22, 35, 97))
    explanation = (
        f"Correlation between mouth-movement activity and speech energy: {corr:.2f}. "
        "Note: low correlation can also arise from dubbing, editing delay, or poor recording."
    )
    return {"name": "cross_modal", "value": {"mouth_speech_correlation": round(corr, 3)},
            "score": score, "confidence": 0.5, "explanation": explanation}


def speech_envelope(audio_path: str, sr: int = 16000) -> dict | None:
    import soundfile as sf
    from app.detectors.audio_forensics import _resample
    try:
        data, file_sr = sf.read(audio_path, dtype="float32")
    except Exception:
        return None
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        data = _resample(data, int(len(data) * sr / file_sr))
        file_sr = sr
    frame = int(0.03 * sr)
    env = np.array([np.sqrt(np.mean(data[i:i + frame] ** 2)) for i in range(0, len(data) - frame, frame)])
    env = env / (env.max() + 1e-6)
    t = np.arange(len(env)) * 0.03
    return {"t": t, "env": env}
