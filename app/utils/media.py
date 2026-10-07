"""Media helpers: audio extraction (via bundled ffmpeg), loading video frames."""
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


def ffmpeg_exe() -> str | None:
    # Prefer the ffmpeg binary bundled with imageio-ffmpeg; fall back to PATH.
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def extract_audio(video_path: str, out_wav: str, sr: int = 16000) -> str | None:
    exe = ffmpeg_exe()
    if exe is None:
        return None
    try:
        subprocess.run(
            [exe, "-y", "-i", str(video_path), "-vn", "-ac", "1",
             "-ar", str(sr), out_wav],
            check=True, capture_output=True, timeout=600,
        )
        return out_wav if Path(out_wav).exists() and Path(out_wav).stat().st_size > 44 else None
    except Exception:
        return None


def read_video_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "duration": float((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / cap.get(cv2.CAP_PROP_FPS))
                        if cap.get(cv2.CAP_PROP_FPS) else 0.0,
        }
    finally:
        cap.release()


def sample_frames(path: str, sample_fps: int, max_frames: int, max_side: int):
    """Yield (timestamp, bgr_frame) downscaled, sampled at sample_fps."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / max(sample_fps, 1))), 1)
    out = []
    idx = 0
    try:
        while len(out) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                h, w = frame.shape[:2]
                scale = min(max_side / max(h, w), 1.0)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                out.append((idx / fps, frame))
            idx += 1
    finally:
        cap.release()
    return out


def load_image(path: str, max_side: int = 1024):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return np.asarray(img)
