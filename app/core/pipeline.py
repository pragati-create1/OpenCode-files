"""Analysis pipeline: dispatch by media type, combine signals, explain."""
from __future__ import annotations

import time
from pathlib import Path

from app.scoring.engine import combine
from app.utils.config import load_config


def _evidence_from_signals(signals: dict) -> list:
    out = []
    for sig in signals.values():
        if isinstance(sig, dict) and sig.get("explanation"):
            score = sig.get("score")
            out.append({
                "signal": sig["name"],
                "score": None if score is None else round(float(score), 1),
                "confidence": sig.get("confidence"),
                "explanation": sig["explanation"],
            })
    return out


def analyze_media(path: str, media_type: str, work_dir: str) -> dict:
    cfg = load_config()
    started = time.time()
    result = {"filename": Path(path).name, "media_type": media_type,
              "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = Path(work_dir) / "processed"
    tmp.mkdir(parents=True, exist_ok=True)

    signals = {}
    details = {}

    if media_type == "image":
        from app.detectors.image_forensics import analyze_image
        from app.models.deepfake_model import deepfake_model_signal
        from app.utils.media import load_image
        im = analyze_image(path, max_side=cfg["image"]["max_side"])
        signals.update(im["signals"])
        rgb = load_image(path, max_side=cfg["image"]["max_side"])
        signals["deepfake_model"] = deepfake_model_signal(rgb)
        details["image_size"] = im["size"]

    elif media_type == "video":
        from app.detectors.video_forensics import analyze_video
        from app.detectors.audio_forensics import analyze_audio
        from app.detectors.cross_modal import cross_modal_signal, speech_envelope
        from app.utils.media import extract_audio
        import cv2
        v = analyze_video(path, cfg["video"]["sample_fps"], cfg["video"]["max_frames"], cfg["video"]["max_side"])
        signals.update(v.get("signals", {}))
        details.update({"video_meta": v.get("meta"), "frames_analyzed": v.get("frames_analyzed"),
                        "suspicious_timestamps": v.get("suspicious_timestamps", []),
                        "frame_scores": v.get("frame_scores", [])})
        if v.get("error"):
            details["error"] = v["error"]
        wav = str(tmp / "extracted_audio.wav")
        audio_path = extract_audio(path, wav, sr=cfg["audio"]["sample_rate"])
        if audio_path:
            a = analyze_audio(audio_path, cfg["audio"]["sample_rate"], cfg["audio"]["max_seconds"])
            signals.update(a.get("signals", {}))
            details["audio"] = {k: v for k, v in a.items() if k != "signals"}
            env = speech_envelope(audio_path, cfg["audio"]["sample_rate"])
            # frames for cross-modal
            from app.utils.media import sample_frames
            frames = sample_frames(path, cfg["video"]["sample_fps"], cfg["video"]["max_frames"], 320)
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in frames]
            times = [t for t, _ in frames]
            signals["cross_modal"] = cross_modal_signal(frames_rgb, times, env)
        else:
            details["audio"] = {"info": "No audio track found or audio extraction unavailable."}

    elif media_type == "audio":
        from app.detectors.audio_forensics import analyze_audio
        a = analyze_audio(path, cfg["audio"]["sample_rate"], cfg["audio"]["max_seconds"])
        signals.update(a.get("signals", {}))
        details["audio"] = {k: v for k, v in a.items() if k != "signals"}

    score_result = combine(signals)
    result.update(score_result)
    result["signals"] = signals
    result["evidence"] = _evidence_from_signals(signals)
    result["suspicious_regions"] = [s["suspicious_region"] for s in signals.values()
                                    if isinstance(s, dict) and s.get("suspicious_region")]
    result["suspicious_timestamps"] = details.get("suspicious_timestamps", details.get("audio", {}).get("suspicious_ranges", []))
    result["details"] = details
    result["methodology"] = [
        "Metadata/EXIF inspection", "JPEG block-artifact analysis", "Regional noise-consistency analysis",
        "Image statistics (sharpness, edges, frequency)", "Face region texture/boundary analysis",
        "Temporal landmark/motion consistency", "Audio spectral forensics", "Audio-visual cross-modal correlation",
    ]
    result["limitations"] = [
        "AI forensic analysis is probabilistic and not definitive legal proof.",
        "Missing metadata does not imply manipulation.",
        "Cross-modal mismatch may result from dubbing or editing rather than fabrication.",
        "No pretrained deepfake checkpoint is bundled by default; model signal is reported as insufficient evidence unless a checkpoint is configured.",
    ]
    result["elapsed_s"] = round(time.time() - started, 2)
    return result
