"""Optional deepfake-model hook.

If PyTorch + a user-supplied checkpoint exist, we run a pretrained classifier.
By default no checkpoint ships with the repo, so the signal reports
'Insufficient evidence' rather than fabricating a prediction.
Place a checkpoint at models/checkpoints/deepfake_classifier.pt with:
  {"state_dict": ..., "arch": "resnet18", "license": "...", "source": "..."}
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_CKPT = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "deepfake_classifier.pt"
_model = None


def _load():
    global _model
    if _model is not None:
        return _model
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)])
        _model = (m, tf, torch)
        return _model
    except Exception:
        _model = False
        return False


def deepfake_model_signal(rgb) -> dict:
    if not _CKPT.exists():
        return {"name": "deepfake_model", "value": {"available": False}, "score": None,
                "confidence": 0.0,
                "explanation": "Insufficient evidence: no pretrained deepfake checkpoint configured "
                               "(models/checkpoints/deepfake_classifier.pt missing).", "suspicious_region": None}
    m = _load()
    if m is False:
        return {"name": "deepfake_model", "value": {"available": False}, "score": None,
                "confidence": 0.0,
                "explanation": "Insufficient evidence: torch/torchvision or checkpoint incompatible.",
                "suspicious_region": None}
    model, tf, torch = m
    x = tf(rgb).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).numpy()[0]
    real_prob = float(p[0])
    return {"name": "deepfake_model", "value": {"available": True, "real_probability": round(real_prob, 3)},
            "score": real_prob * 100, "confidence": float(abs(real_prob - 0.5) * 2),
            "explanation": f"Pretrained classifier 'real' probability {real_prob:.2f}.",
            "suspicious_region": None}
