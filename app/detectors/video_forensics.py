"""Video forensics: frame sampling, per-frame stats, temporal consistency."""
from __future__ import annotations

import cv2
import numpy as np

from app.detectors.image_forensics import detect_faces, _gray_small
from app.utils.media import sample_frames, read_video_meta


def _frame_sharpness(frame_bgr) -> float:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def analyze_video(path: str, sample_fps: int = 2, max_frames: int = 120, max_side: int = 640) -> dict:
    meta = read_video_meta(path)
    frames = sample_frames(path, sample_fps, max_frames, max_side)
    if not frames:
        return {"error": "Could not read frames from video.", "signals": {}, "frames": []}

    timestamps, sharp, motion, face_centers, face_counts = [], [], [], [], []
    prev_gray, prev_faces = None, None
    suspicious = []
    for i, (t, frame) in enumerate(frames):
        g = _gray_small(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sharp.append(_frame_sharpness(frame))
        faces = detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_counts.append(len(faces))
        if faces:
            cx = np.mean([x + w / 2 for x, _, w, _ in faces])
            cy = np.mean([y + h / 2 for _, y, _, h in faces])
            face_centers.append((cx, cy))
        else:
            face_centers.append(None)
        if prev_gray is not None:
            motion.append(float(np.abs(g.astype(float) - prev_gray.astype(float)).mean()))
        prev_gray = g
        timestamps.append(round(t, 3))

    # Temporal landmark/centroid consistency: large jumps are flagged.
    jumps = []
    for i in range(1, len(face_centers)):
        a, b = face_centers[i - 1], face_centers[i]
        if a is None or b is None:
            jumps.append(np.nan)
        else:
            diag = max(frames[i][1].shape[0], frames[i][1].shape[1])
            jumps.append(float(np.hypot(a[0] - b[0], a[1] - b[1]) / diag))
    jumps_arr = np.array([j for j in jumps if not np.isnan(j)]) if jumps else np.array([0.0])
    max_jump = float(jumps_arr.max()) if jumps_arr.size else 0.0
    mean_jump = float(jumps_arr.mean()) if jumps_arr.size else 0.0

    motion_arr = np.array(motion) if motion else np.array([0.0])
    motion_cv = float(motion_arr.std() / (motion_arr.mean() + 1e-6))

    sharp_arr = np.array(sharp)
    sharp_cv = float(sharp_arr.std() / (sharp_arr.mean() + 1e-6))

    temporal_score = 88.0
    notes = []
    if max_jump > 0.35:
        temporal_score -= 35; notes.append(f"abrupt face displacement ({max_jump:.2f} of frame)")
    elif max_jump > 0.20:
        temporal_score -= 15; notes.append("moderate face displacement")
    if motion_cv > 2.5:
        temporal_score -= 20; notes.append("very irregular inter-frame change")
    if sharp_cv > 1.5:
        temporal_score -= 15; notes.append("large sharpness fluctuation between frames")
    if int(np.max(face_counts)) - int(np.min(face_counts)) > 1 and np.mean(face_counts) > 0:
        temporal_score -= 10; notes.append("face count fluctuated across frames")
    temporal_score = float(np.clip(temporal_score, 20, 96))

    # While detecting sudden local changes, mark suspicious timestamps.
    if motion:
        thr = motion_arr.mean() + 2.5 * motion_arr.std()
        for i, m in enumerate(motion):
            if m > thr:
                suspicious.append({"t": timestamps[i + 1], "frame_index": i + 1, "reason": "sudden inter-frame change", "magnitude": round(m, 2)})
    for i, j in enumerate(jumps):
        if not np.isnan(j) and j > 0.25:
            suspicious.append({"t": timestamps[i + 1], "frame_index": i + 1, "reason": "abrupt face displacement", "magnitude": round(float(j), 3)})

    # frame duplication check
    dup_pairs = 0
    grays = []
    for _, f in frames:
        grays.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(int))
    for i in range(1, len(grays)):
        if np.abs(grays[i] - grays[i - 1]).mean() < 0.8:
            dup_pairs += 1
    if dup_pairs > max(3, len(grays) // 3):
        temporal_score = max(15.0, temporal_score - 20)
        notes.append(f"{dup_pairs} near-duplicate consecutive frames")

    signals = {
        "temporal": {
            "name": "temporal", "value": {"frames_analyzed": len(frames), "max_face_jump": round(max_jump, 3),
                                          "mean_face_jump": round(mean_jump, 3), "motion_cv": round(motion_cv, 2),
                                          "sharpness_cv": round(sharp_cv, 2), "duplicate_frame_pairs": dup_pairs},
            "score": temporal_score, "confidence": 0.6,
            "explanation": "Temporal consistency: " + ("; ".join(notes) if notes else "no abrupt inconsistencies detected") + ".",
        },
    }
    return {
        "signals": signals,
        "meta": meta,
        "frames_analyzed": len(frames),
        "suspicious_timestamps": suspicious,
        "frame_scores": [{"t": timestamps[i], "sharpness": round(sharp[i], 1),
                          "faces": face_counts[i]} for i in range(len(timestamps))],
        "timestamps": timestamps,
    }
