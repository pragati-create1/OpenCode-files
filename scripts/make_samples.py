"""Generate small synthetic DEMO samples (labelled as synthetic).

These are NOT real media — they exist only to exercise the pipeline.
"""
from pathlib import Path

import numpy as np
import cv2
import soundfile as sf

OUT = Path("data/samples")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

# 1) A synthetic 'natural-ish' image with texture
img = np.zeros((512, 512, 3), np.uint8)
img[:] = (200, 180, 160)
cv2.circle(img, (256, 220), 90, (190, 150, 130), -1)          # face-like blob
cv2.circle(img, (222, 200), 10, (40, 40, 40), -1)
cv2.circle(img, (290, 200), 10, (40, 40, 40), -1)
cv2.ellipse(img, (256, 280), (30, 12), 0, 0, 180, (120, 60, 60), 2)
noise = rng.normal(0, 6, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT / "sample_image.png"), img)

# 2) Synthetic audio (tone bursts with silence)
sr = 16000
t = np.arange(sr * 2) / sr
tone = 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
sf.write(OUT / "sample_audio.wav", tone.astype(np.float32), sr)

# 3) Synthetic video with moving blob + beep audio
w, h, fps, secs = 320, 240, 10, 3
vw = cv2.VideoWriter(str(OUT / "sample_video_silent.avi"),
                     cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
for i in range(fps * secs):
    frame = np.full((h, w, 3), 180, np.uint8)
    cv2.circle(frame, (40 + i * 8, 120), 25, (90, 130, 200), -1)
    vw.write(frame)
vw.release()

print(f"Samples written to {OUT.resolve()}")
