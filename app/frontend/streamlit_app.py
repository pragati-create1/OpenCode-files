"""Streamlit dashboard for HNX26PSI10."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.set_page_config(page_title="HNX26PSI10 · Deepfake Forensics", layout="wide")

from app.core.pipeline import analyze_media
from app.core.report import to_json, to_html, save_report
from app.utils.validation import detect_media_type, ValidationError

PAGES = ["Home", "Upload & Analyze", "Analysis Results", "Evidence",
         "Suspicious Frames", "Audio Analysis", "Cross-Modal Analysis",
         "Technical Details", "Export Report"]

page = st.sidebar.radio("Navigation", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("AI forensics is probabilistic — not legal proof.")

if "result" not in st.session_state:
    st.session_state.result = None


def _verdict_badge(v):
    c = {"Likely Real": "🟢", "Uncertain": "🟡", "Likely Manipulated": "🔴"}.get(v, "⚪")
    return f"{c} {v}"


if page == "Home":
    st.title("🛡️ HNX26PSI10 — Multimodal Deepfake & Digital Forensics")
    st.write("Analyze images, videos, and audio for signs of manipulation using "
             "multiple independent forensic signals, transparent scoring, and explainable evidence.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Signals", "9")
    col2.metric("Media types", "Image · Video · Audio")
    col3.metric("Exports", "JSON · HTML")
    st.info("Go to **Upload & Analyze** to begin. Demo samples can be generated with "
            "`python scripts/make_samples.py`.")

elif page == "Upload & Analyze":
    st.header("Upload & Analyze")
    up = st.file_uploader("Upload media", type=["jpg", "jpeg", "png", "bmp", "webp",
                                                "mp4", "avi", "mov", "mkv", "webm",
                                                "wav", "mp3", "ogg", "flac", "m4a", "aac"])
    mode = st.selectbox("Analysis mode", ["Full forensic analysis", "Quick (image signals only for images)"])
    if st.button("Start forensic analysis", disabled=up is None):
        if up is None:
            st.warning("Choose a file first.")
        else:
            tmp = Path(tempfile.mkdtemp()) / up.name
            tmp.write_bytes(up.getbuffer())
            try:
                media_type = detect_media_type(tmp)
            except ValidationError as e:
                st.error(str(e)); st.stop()
            with st.spinner("Running forensic pipeline…"):
                try:
                    st.session_state.result = analyze_media(str(tmp), media_type, work_dir=".")
                except Exception as e:
                    st.error(f"Analysis failed: {type(e).__name__}: {e}"); st.stop()
            st.success("Analysis complete. See the other pages.")

res = st.session_state.result

if res and page in ("Analysis Results", "Evidence", "Suspicious Frames", "Audio Analysis",
                    "Cross-Modal Analysis", "Technical Details", "Export Report"):
    st.header(page)
    c1, c2, c3 = st.columns(3)
    c1.metric("Authenticity Score", f"{res.get('authenticity_score')} / 100" if res.get('authenticity_score') is not None else "N/A")
    c2.metric("Verdict", res.get("verdict", "—"))
    c3.metric("Confidence", f"{int((res.get('confidence') or 0) * 100)}%")

    if page == "Analysis Results":
        st.subheader("Signal scores")
        rows = []
        for k, s in res.get("signals", {}).items():
            if isinstance(s, dict):
                rows.append({"signal": k, "score": s.get("score"),
                             "confidence": s.get("confidence"),
                             "explanation": s.get("explanation", "")})
        st.dataframe(rows, use_container_width=True)
        for r in rows:
            if r["score"] is not None:
                st.progress(min(max(r["score"] / 100, 0.0), 1.0), text=f"{r['signal']}: {r['score']:.0f}/100")

    if page == "Evidence":
        for e in res.get("evidence", []):
            icon = "✅" if (e.get("score") or 0) >= 65 else ("⚠️" if (e.get('score') or 100) > 40 else "🚩")
            st.write(f"{icon} **{e['signal']}** — {e['explanation']}")

    if page == "Suspicious Frames":
        ts = res.get("suspicious_timestamps") or []
        if ts:
            st.table(ts)
        else:
            st.info("No suspicious timestamps flagged.")

    if page == "Audio Analysis":
        audio = res.get("details", {}).get("audio")
        if audio and "error" in audio:
            st.warning(audio["error"])
        elif audio:
            st.json(audio)
        else:
            st.info("No audio analyzed.")

    if page == "Cross-Modal Analysis":
        cm = res.get("signals", {}).get("cross_modal")
        if cm:
            st.write(f"**Score:** {cm.get('score')}/100 — {cm.get('explanation')}")
            st.caption("Audio-video mismatch alone does not imply fabrication (dubbing/editing/sync errors).")
        else:
            st.info("Cross-modal analysis requires a video with audio.")

    if page == "Technical Details":
        st.json(res.get("details", {}))
        st.subheader("Methodology")
        for m in res.get("methodology", []):
            st.write(f"• {m}")

    if page == "Export Report":
        st.download_button("Download JSON report", to_json(res), "report.json", "application/json")
        st.download_button("Download HTML report", to_html(res), "report.html", "text/html")
        if st.button("Save report to disk"):
            paths = save_report(res, "data/results", Path(res.get("filename", "report")).stem)
            st.success(f"Saved: {paths['json']} and {paths['html']}")
elif page in PAGES[2:] and not res:
    st.warning("No analysis yet — upload a file first.")
