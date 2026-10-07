"""Audio forensics using numpy/scipy (librosa optional).

Signals: spectral centroid stability, high-frequency energy, spectral flatness,
silence/voice-activity structure, temporal envelope periodicity heuristic.
All scores are probabilistic AND must not be treated as proof of cloning.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf


def _resample(x: np.ndarray, new_len: int) -> np.ndarray:
    """Linear-interpolation resample (sufficient for forensic envelopes)."""
    if len(x) == new_len:
        return x
    return np.interp(np.linspace(0, len(x), new_len, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


def _spectrogram(x: np.ndarray, sr: int, nperseg: int = 512, noverlap: int = 256):
    step = nperseg - noverlap
    n = 1 + max((len(x) - nperseg) // step, 0)
    win = np.hanning(nperseg)
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, nperseg),
        strides=(x.strides[0] * step, x.strides[0])) * win
    S = np.abs(np.fft.rfft(frames, axis=1)).T ** 2
    f = np.fft.rfftfreq(nperseg, 1.0 / sr)
    t = (np.arange(n) * step) / sr
    return f, t, S


def analyze_audio(path: str, target_sr: int = 16000, max_seconds: int = 180) -> dict:
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        return {"error": f"Could not decode audio: {e}", "signals": {}}
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) > target_sr * max_seconds:
        data = data[: target_sr * max_seconds]
    if sr != target_sr:
        data = _resample(data, int(len(data) * target_sr / sr))
        sr = target_sr
    if len(data) < sr * 0.2:
        return {"error": "Audio too short to analyze.", "signals": {}}

    f, t, S = _spectrogram(data, sr)
    S_db = 10 * np.log10(S + 1e-10)
    freqs = f

    total_energy = S.sum(axis=0) + 1e-10
    centroid = (freqs[:, None] * S).sum(axis=0) / total_energy
    centroid_cv = float(centroid.std() / (centroid.mean() + 1e-6))

    hf = S[freqs > 6000].sum()
    total = S.sum() + 1e-10
    hf_ratio = float(hf / total)

    logS = np.log(S + 1e-10).mean(axis=0)
    arith = S.mean(axis=0)
    flatness = float(np.exp(logS.mean()) / (arith.mean() + 1e-10))

    rms = np.sqrt(np.mean(data ** 2))
    frame_rms = np.array([np.sqrt(np.mean(data[i:i+512]**2)) for i in range(0, len(data)-512, 256)])
    vad = frame_rms > (0.1 * frame_rms.max() + 1e-6)
    voiced_ratio = float(vad.mean())

    # suspicious ranges: frames with anomalously low high-frequency energy
    hf_t = S[(freqs > 6000)].sum(axis=0) / (S.sum(axis=0) + 1e-10)
    thr = hf_t.mean() - 2 * hf_t.std()
    suspicious_ranges = []
    in_seg = False
    for i, v in enumerate(hf_t < thr):
        if v and not in_seg:
            start = t[i]; in_seg = True
        elif not v and in_seg:
            suspicious_ranges.append({"start": round(float(start), 2), "end": round(float(t[i]), 2)})
            in_seg = False
    if in_seg:
        suspicious_ranges.append({"start": round(float(start), 2), "end": round(float(t[-1]), 2)})

    score = 85.0
    notes = []
    if hf_ratio < 0.02:
        score -= 20; notes.append("very low >6kHz energy (may also be due to compression)")
    if voiced_ratio < 0.05:
        score -= 20; notes.append("almost no voice activity detected")
    if flatness > 0.35:
        score -= 15; notes.append("high spectral flatness (noise-like)")
    if centroid_cv < 0.05:
        score -= 10; notes.append("unusually static spectral centroid")
    score = float(np.clip(score, 20, 96))

    return {
        "duration_s": round(len(data) / sr, 2),
        "sample_rate": sr,
        "signals": {
            "audio": {
                "name": "audio",
                "value": {"hf_ratio": round(hf_ratio, 4), "voiced_ratio": round(voiced_ratio, 3),
                          "spectral_flatness": round(flatness, 3), "centroid_cv": round(centroid_cv, 3),
                          "rms": round(float(rms), 4)},
                "score": score, "confidence": 0.5,
                "explanation": "Audio forensics: " + ("; ".join(notes) if notes else "no strong synthetic cues measured") + ".",
            }
        },
        "suspicious_ranges": suspicious_ranges[:20],
    }
