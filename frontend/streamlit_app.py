# app/streamlit_app.py

"""
Streamlit Frontend for PDD Agent.
Thin UI layer that communicates with the Backend FastAPI service.
Two modes:
1. Meeting Recording (Audio+Video) — transcript optional, auto-generated if missing
2. Silent Screen Recording (Video Only) — vision-based analysis
"""

import streamlit as st
import requests
import time
import os
import dotenv



# Backend API base URL (configurable via environment variable)
dotenv.load_dotenv()
BACKEND_URL = os.getenv("PDD_BACKEND_URL", "http://localhost:8001")


def _backend_health():
    """Check backend API health."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=120)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        return None
    return None


def _submit_pipeline(
    video_bytes,
    video_filename,
    input_mode,
    project_name=None,
    document_type="PDD",
    whisper_model="base",
    transcript_text=None,
    transcript_bytes=None,
    transcript_filename=None,
    ssim_threshold=0.92,
    max_frames=60,
    enable_micro_frames=True,
    annotate=True,
    pii_redaction=True,
):
    """Submit a pipeline processing job to the backend."""
    url = f"{BACKEND_URL}/api/pipeline/process"
    files = {"video": (video_filename, video_bytes, "video/mp4")}
    if transcript_bytes and transcript_filename:
        files["transcript_file"] = (transcript_filename, transcript_bytes, "text/plain")

    data = {
        "input_mode": input_mode,
        "document_type": document_type,
        "whisper_model": whisper_model,
        "ssim_threshold": str(ssim_threshold),
        "max_frames": str(max_frames),
        "enable_micro_frames": str(enable_micro_frames).lower(),
        "annotate": str(annotate).lower(),
        "pii_redaction": str(pii_redaction).lower(),
    }
    if project_name:
        data["project_name"] = project_name
    if transcript_text:
        data["transcript_text"] = transcript_text

    resp = requests.post(url, files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _poll_job_status(job_id):
    """Poll the backend for job status."""
    url = f"{BACKEND_URL}/api/pipeline/status/{job_id}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _download_document(job_id):
    """Download the generated document from the backend."""
    url = f"{BACKEND_URL}/api/pipeline/download/{job_id}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def main():
    st.set_page_config(
        page_title="PDD Generation Agent",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 PDD Generation Agent")
    st.markdown(
        "Convert meeting recordings or silent screen recordings into "
        "comprehensive Process Definition Documents."
    )

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔌 Backend Connection")
        health = _backend_health()
        if health:
            st.success("✓ Backend API Connected")
            if health.get("gemini_available"):
                st.success("✓ Gemini API Connected")
                st.caption(f"Model: `{health.get('gemini_model', 'N/A')}`")
            elif health.get("gemini_configured"):
                st.warning("⚠ Gemini configured but health-check failed")
                if health.get("gemini_error"):
                    st.caption(f"Error: {health['gemini_error']}")
            else:
                st.error("✗ Gemini API Not Connected")
                st.caption("Set GEMINI_API_KEY in backend environment.")
        else:
            st.error(f"✗ Backend API not reachable at {BACKEND_URL}")
            st.caption("Start the backend: `uvicorn backend.main:app --port 8001`")

        st.markdown("---")

        input_mode = st.radio(
            "📥 Input Type",
            [
                "🗣️ Meeting Recording (Audio+Video)",
                "🔇 Silent Screen Recording (Video Only)",
            ],
            index=0,
            help="Meeting Recording: has audio (or provide transcript). "
                 "Silent Recording: no audio, vision-based analysis.",
        )

        st.markdown("---")

        doc_type_choice = st.selectbox(
            "📋 Document Type",
            [
                "PDD - Process Definition Document",
                "BRD - Business Requirements Document",
            ],
            index=0,
        )
        document_type = "PDD" if doc_type_choice.startswith("PDD") else "BRD"

        if input_mode == "🗣️ Meeting Recording (Audio+Video)":
            st.markdown("---")
            st.subheader("🎙️ Audio Settings")
            whisper_model = st.selectbox(
                "Whisper Model",
                ["base", "small", "medium", "large"],
                index=0,
                help="Used only if no transcript is provided.",
            )
        else:
            whisper_model = "base"

        if input_mode == "🔇 Silent Screen Recording (Video Only)":
            st.markdown("---")
            st.subheader("🎬 Video Settings")

            ssim_threshold = st.slider(
                "Scene Sensitivity",
                min_value=0.70,
                max_value=0.95,
                value=0.92,
                step=0.05,
            )

            max_frames = st.number_input(
                "Max Key Frames",
                min_value=10,
                max_value=150,
                value=60,
                step=10,
            )

            enable_micro_frames = st.checkbox("Enable Micro-frames", value=True)
            annotate = st.checkbox("Annotate Screenshots", value=True)
        else:
            ssim_threshold = 0.92
            max_frames = 60
            enable_micro_frames = False
            annotate = True

        st.markdown("---")
        st.subheader("🔒 Privacy")
        pii_redaction = st.checkbox("Enable PII Redaction", value=True)

    # ── Main Content ──
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Input")

        project_name = st.text_input(
            "Project Name",
            placeholder="e.g., Monthly Report Generation",
            help="Required for silent videos. Auto-detected for meetings if left blank.",
        )

        st.subheader("🎬 Video File")
        video_file = st.file_uploader(
            "Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"]
        )
        if video_file:
            st.video(video_file)
            size_mb = len(video_file.getvalue()) / (1024 * 1024)
            st.caption(f"📏 {size_mb:.1f} MB")

        transcript_text = None
        transcript_bytes = None
        transcript_filename = None

        if input_mode == "🗣️ Meeting Recording (Audio+Video)":
            st.subheader("📝 Transcript (Optional)")
            transcript_input_method = st.radio(
                "Transcript Input",
                ["None (Auto-transcribe)", "Upload File", "Paste Text"],
                index=0,
            )

            if transcript_input_method == "Upload File":
                transcript_file = st.file_uploader(
                    "Upload Transcript", type=["txt", "srt", "vtt"]
                )
                if transcript_file:
                    transcript_bytes = transcript_file.read()
                    transcript_filename = transcript_file.name
                    transcript_file.seek(0)
                    preview = transcript_bytes.decode("utf-8", errors="replace")
                    with st.expander("Preview", expanded=False):
                        st.text(preview[:1000] + ("..." if len(preview) > 1000 else ""))

            elif transcript_input_method == "Paste Text":
                transcript_text = st.text_area(
                    "Paste Transcript",
                    height=200,
                    placeholder="Paste your transcript here...",
                )

    with col2:
        st.header("📊 Output")
        output_placeholder = st.empty()
        status_placeholder = st.empty()
        download_placeholder = st.empty()

    # ── Validation ──
    can_process = False
    missing = ""

    if input_mode == "🗣️ Meeting Recording (Audio+Video)":
        can_process = bool(video_file)
        missing = "Upload a meeting recording video."
    elif input_mode == "🔇 Silent Screen Recording (Video Only)":
        can_process = bool(video_file) and bool(project_name.strip())
        if not project_name.strip():
            missing = "Project Name is required for silent videos."
        elif not video_file:
            missing = "Upload a silent screen recording."

    # ── Process Button ──
    st.markdown("---")

    if st.button(
        f"🚀 Generate {document_type}",
        type="primary",
        use_container_width=True,
    ):
        if not can_process:
            st.error(missing)
            return

        if not health:
            st.error("Backend API is not reachable. Please start the backend service.")
            return

        if not health.get("gemini_available"):
            st.error("Gemini API is not available. Check backend configuration.")
            return

        mode_key = "meeting" if "Meeting" in input_mode else "silent"

        with st.spinner("Submitting job to backend..."):
            try:
                result = _submit_pipeline(
                    video_bytes=video_file.getvalue(),
                    video_filename=video_file.name,
                    input_mode=mode_key,
                    project_name=project_name.strip() if project_name else None,
                    document_type=document_type,
                    whisper_model=whisper_model,
                    transcript_text=transcript_text,
                    transcript_bytes=transcript_bytes,
                    transcript_filename=transcript_filename,
                    ssim_threshold=ssim_threshold,
                    max_frames=max_frames,
                    enable_micro_frames=enable_micro_frames,
                    annotate=annotate,
                    pii_redaction=pii_redaction,
                )
                job_id = result.get("job_id")
            except requests.HTTPError as e:
                st.error(f"Backend error: {e.response.text}")
                return
            except Exception as e:
                st.error(f"Failed to submit job: {e}")
                return

        status_placeholder.info(f"🔄 Processing... Job ID: {job_id}")

        # Poll for completion
        while True:
            time.sleep(3)
            try:
                status = _poll_job_status(job_id)
            except Exception:
                continue

            job_status = status.get("status", "")

            if job_status == "processing":
                status_placeholder.info(f"🔄 Processing... ({status.get('message', '')})")
            elif job_status == "completed":
                output_placeholder.success(f"✅ {document_type} Generated Successfully!")
                status_placeholder.empty()

                try:
                    doc_bytes = _download_document(job_id)
                    download_placeholder.download_button(
                        label=f"📥 Download {document_type} Document",
                        data=doc_bytes,
                        file_name=status.get("document_filename", f"{document_type}.docx"),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Failed to download document: {e}")
                break
            elif job_status == "failed":
                output_placeholder.error(
                    f"❌ Generation failed: {status.get('message', 'Unknown error')}"
                )
                status_placeholder.empty()
                break
            else:
                status_placeholder.info(f"⏳ Status: {job_status}")


if __name__ == "__main__":
    main()