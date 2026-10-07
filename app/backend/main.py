"""FastAPI backend for HNX26PSI10."""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.core.pipeline import analyze_media
from app.utils.config import load_config
from app.utils.validation import validate_upload, safe_filename, ValidationError

cfg = load_config()
UPLOAD_DIR = Path(cfg["uploads"]["dir"]); UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(cfg["uploads"]["max_bytes"])

app = FastAPI(title="HNX26PSI10 Forensics API", version="1.0.0")
_cache: dict[str, dict] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    name = safe_filename(file.filename or "upload")
    uid = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{uid}_{name}"
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        media_type = validate_upload(dest, MAX_BYTES)
    except ValidationError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    try:
        result = analyze_media(str(dest), media_type, work_dir=str(Path(cfg["results"]["dir"]).parents[1]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}: {e}")
    result["id"] = uid
    _cache[uid] = result
    return JSONResponse(result)


@app.get("/result/{uid}")
def result(uid: str):
    if uid not in _cache:
        raise HTTPException(404, "Result not found.")
    return _cache[uid]
