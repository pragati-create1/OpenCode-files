"""Upload validation: extension allow-lists, size limits, magic bytes."""
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}

_MAGIC = {
    b"\xff\xd8": "image",
    b"\x89PNG": "image",
    b"BM": "image",
    b"RIFF": "media",   # wav / avi / webp
    b"ID3": "audio",
    b"\x1a\x45\xdf\xa3": "video",  # webm/mkv
    b"ftyp": "media",
}


class ValidationError(Exception):
    pass


def detect_media_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    raise ValidationError(f"Unsupported file extension: {ext or '(none)'}")


def validate_upload(path: str | Path, max_bytes: int) -> str:
    p = Path(path)
    if not p.exists():
        raise ValidationError("File not found.")
    size = p.stat().st_size
    if size == 0:
        raise ValidationError("File is empty.")
    if size > max_bytes:
        raise ValidationError(f"File exceeds size limit ({max_bytes // (1024*1024)} MB).")
    with open(p, "rb") as f:
        head = f.read(16)
    if not any(head.startswith(m) or head[4:8] == m for m in _MAGIC):
        raise ValidationError("File content does not match a supported media signature.")
    return detect_media_type(p)


def safe_filename(name: str) -> str:
    """Strip directories and unsafe characters; never trust client paths."""
    base = Path(name).name
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in base]
    out = "".join(keep).lstrip(".")
    return out or "upload"
