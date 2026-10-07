# HNX26PSI10 — Multimodal Deepfake & Digital Forensics

An AI-powered digital-forensics platform that analyzes images, videos, and audio
to determine whether media is authentic or manipulated. It combines multiple
independent forensic signals (never one detector) and produces an authenticity
score, verdict, confidence, per-signal scores, evidence, suspicious regions and
timestamps, audio findings, cross-modal analysis, and exportable reports.

> ⚠️ AI forensic analysis is probabilistic. Results are NOT definitive legal proof.

## Architecture

```
Upload → Validation → Media pipeline → Detectors (signals) → Scoring engine → Report
                                     ├─ image forensics (metadata, JPEG, noise, stats, face)
                                     ├─ video (frames, temporal, duplicates)
                                     ├─ audio (spectral forensics)
                                     ├─ cross-modal (mouth vs speech envelope)
                                     └─ optional deepfake model hook (checkpoint-based)
```

Modular packages under `app/`: `detectors/`, `models/`, `scoring/`, `core/`,
`backend/` (FastAPI), `frontend/` (Streamlit), `utils/`.

## Installation (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

Dashboard (recommended):
```powershell
streamlit run app/frontend/streamlit_app.py
```

REST API:
```powershell
uvicorn app.backend.main:app --port 8000
```

## Tests

```powershell
python -m pytest -q
```

## Demo & utility scripts

```powershell
python scripts/make_samples.py                 # synthetic demo media (labelled synthetic)
python scripts/test_robustness.py <image>      # compression/resize/crop/noise score drift
python scripts/evaluate.py                     # metrics over data/eval (if labelled data present)
```

## Forensic signals

| Signal | What it measures |
|---|---|
| metadata | EXIF presence, software tags, dimensions, format |
| compression | JPEG block-boundary discontinuity, quantization tables |
| noise | regional high-pass residual consistency (CV) |
| image_stats | sharpness (Laplacian var), edge density, HF energy ratio |
| face | face texture sharpness, boundary gradient |
| temporal | face-centroid jumps, motion irregularity, sharpness CV, duplicate frames |
| audio | HF energy ratio, voiced ratio, spectral flatness, centroid stability |
| cross_modal | correlation of mouth activity vs speech-energy envelope |
| deepfake_model | pretrained classifier — reports "insufficient evidence" unless you supply `models/checkpoints/deepfake_classifier.pt` (see `app/models/deepfake_model.py`) |

## Scoring

Weighted average of signal scores (0–100) with weights in `config/settings.yaml`,
renormalized over signals that provided actual evidence and weighted by each
signal's confidence. Verdict thresholds: ≥68 Likely Real, ≤38 Likely Manipulated,
otherwise **Uncertain**. Weak/conflicting evidence yields reduced confidence.

## Models

No pretrained weights are bundled. To enable the deepfake model signal, place a
PyTorch checkpoint (`resnet18` backbone trained for real/fake) at
`models/checkpoints/deepfake_classifier.pt` and install `torch`+`torchvision`.
License and provenance must be documented by the person supplying the checkpoint.

## Datasets & evaluation

Prepare data in `data/eval/real/...` and `data/eval/fake/...` (images, videos,
audio). Do **not** commit huge datasets. Supported formats: common image/video/audio
types. Run `python scripts/evaluate.py` to compute accuracy/precision/recall/F1/
ROC-AUC/FPR/FNR/confusion matrix. If no labelled data exists it prints
"Evaluation not yet performed."

## Robustness

`scripts/test_robustness.py` compresses (Q50), resizes (50%), crops, and adds
noise to an image, re-analyzes each variant, and reports score drifts and verdict
changes. Re-encoding/compression are treated as expected artifacts, not proof of
fake.

## Limitations

- Heuristic forensic signals are prototypes; thresholds may need tuning per domain.
- Face analysis uses Haar cascades (frontal, well-lit faces).
- Cross-modal mismatch can be caused by dubbing/editing/sync error — it is only
  one piece of evidence.
- Very short/long media is sampled; extreme cases return "insufficient evidence".
- Audio spectrograms use a lightweight numpy STFT implementation for
  portability on Python 3.14.

## Future improvements

- Integrate a pretrained deepfake checkpoint + Grad-CAM heatmaps.
- Robust audio spoofing model (e.g., wav2vec2-based) with license-reviewed weights.
- Face-mesh landmarks for finer temporal geometry.
- GPU batching and progress streaming in the API.
- PDF report export.
