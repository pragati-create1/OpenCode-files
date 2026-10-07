# Usage Guide

1. `pip install -r requirements.txt`
2. `streamlit run app/frontend/streamlit_app.py`
3. Upload media → Start analysis → review pages (Results, Evidence, Frames, Audio, Cross-modal, Export).
4. CLI: `uvicorn app.backend.main:app` then `POST /analyze` with a multipart file.
5. Reports: JSON + HTML saved under `data/results/`.
6. Robustness: `python scripts/test_robustness.py data/samples/sample_image.png`
7. Evaluation: place labelled samples under `data/eval/{real,fake}/...`, then `python scripts/evaluate.py`
