from pathlib import Path

import numpy as np
import cv2
import pytest


@pytest.fixture(scope="session")
def sample_image(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "img.png"
    img = np.full((256, 256, 3), 170, np.uint8)
    cv2.circle(img, (128, 128), 60, (200, 170, 150), -1)
    noise = np.random.default_rng(0).normal(0, 5, img.shape)
    cv2.imwrite(str(p), np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    return str(p)


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory):
    import soundfile as sf
    p = tmp_path_factory.mktemp("m") / "a.wav"
    sr = 16000
    t = np.arange(sr) / sr
    tone = 0.2 * np.sin(2 * np.pi * 200 * t)
    sf.write(str(p), tone.astype(np.float32), sr)
    return str(p)
