"""
Pipeline routes — submit video processing jobs and check status.
"""

import os
import uuid
import tempfile
import threading
import logging
from typing import Dict

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from api.models import (
    InputMode,
    DocumentType,
    PipelineResponse,
    PipelineStatus,
    JobStatusResponse,
)
from core.config import config
from core.gemini_client import gemini_client

from pipeline.audio_pipeline import AudioPipeline
from pipeline.video_pipeline import VideoPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# In-memory job store (replace with DB/Redis for production)
_jobs: Dict[str, dict] = {}


def _run_pipeline(job_id: str, params: dict):
    """Execute pipeline in background thread."""
    try:
        _jobs[job_id]["status"] = PipelineStatus.PROCESSING

        input_mode = params["input_mode"]
        video_path = params["video_path"]
        output_dir = params["output_dir"]
        project_name = params.get("project_name")
        document_type = params.get("document_type", "PDD")

        # Apply document type config
        if document_type == "BRD":
            config.document.document_type = "BRD"
            config.document.document_type_full = "Business Requirements Document"
        else:
            config.document.document_type = "PDD"
            config.document.document_type_full = "Process Definition Document"

        config.redaction.enabled = params.get("pii_redaction", True)

        result_path = None

        if input_mode == InputMode.MEETING:
            transcript_path = params.get("transcript_path")
            whisper_model = params.get("whisper_model", "base")
            agent = AudioPipeline(output_dir)
            result_path = agent.process(
                video_path=video_path,
                project_name=project_name,
                whisper_model=whisper_model,
                transcript_path=transcript_path,
            )

        elif input_mode == InputMode.SILENT:
            agent = VideoPipeline(output_dir)
            result_path = agent.process(
                video_path=video_path,
                project_name=project_name or "Unnamed Project",
                ssim_threshold=params.get("ssim_threshold", 0.92),
                max_frames=params.get("max_frames", 60),
                annotate=params.get("annotate", True),
                enable_micro_frames=params.get("enable_micro_frames", True),
            )

        if result_path and os.path.exists(result_path):
            _jobs[job_id]["status"] = PipelineStatus.COMPLETED
            _jobs[job_id]["document_path"] = result_path
            _jobs[job_id]["document_filename"] = os.path.basename(result_path)
            _jobs[job_id]["project_name"] = project_name
            _jobs[job_id]["message"] = "Document generated successfully."
            logger.info("Job %s completed: %s", job_id, result_path)
        else:
            _jobs[job_id]["status"] = PipelineStatus.FAILED
            _jobs[job_id]["message"] = "Pipeline completed but no document was generated."

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _jobs[job_id]["status"] = PipelineStatus.FAILED
        _jobs[job_id]["message"] = f"Pipeline error: {str(e)}"


@router.post("/process", response_model=PipelineResponse)
async def start_pipeline(
    video: UploadFile = File(...),
    input_mode: str = Form(...),
    project_name: str = Form(None),
    document_type: str = Form("PDD"),
    whisper_model: str = Form("base"),
    transcript_text: str = Form(None),
    ssim_threshold: float = Form(0.92),
    max_frames: int = Form(60),
    enable_micro_frames: bool = Form(True),
    annotate: bool = Form(True),
    pii_redaction: bool = Form(True),
    transcript_file: UploadFile = File(None),
):
    """Submit a video for PDD/BRD generation."""

    if not gemini_client.is_configured():
        raise HTTPException(status_code=503, detail="Gemini API not configured.")

    # Validate input_mode
    try:
        mode = InputMode(input_mode)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid input_mode: {input_mode}. Use 'meeting' or 'silent'."
        )

    job_id = uuid.uuid4().hex[:12]
    work_dir = os.path.join(tempfile.gettempdir(), "pdd_jobs", job_id)
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save uploaded video
    video_path = os.path.join(work_dir, video.filename or "upload.mp4")
    with open(video_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # Save transcript if provided
    transcript_path = None
    if transcript_file:
        transcript_path = os.path.join(work_dir, transcript_file.filename or "transcript.txt")
        with open(transcript_path, "wb") as f:
            f.write(await transcript_file.read())
    elif transcript_text and transcript_text.strip():
        transcript_path = os.path.join(work_dir, "pasted_transcript.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

    params = {
        "input_mode": mode,
        "video_path": video_path,
        "output_dir": output_dir,
        "project_name": project_name,
        "document_type": document_type,
        "whisper_model": whisper_model,
        "transcript_path": transcript_path,
        "ssim_threshold": ssim_threshold,
        "max_frames": max_frames,
        "enable_micro_frames": enable_micro_frames,
        "annotate": annotate,
        "pii_redaction": pii_redaction,
    }

    _jobs[job_id] = {
        "status": PipelineStatus.PENDING,
        "message": "Job submitted.",
        "document_path": None,
        "document_filename": None,
        "project_name": project_name,
        "work_dir": work_dir,
    }

    thread = threading.Thread(target=_run_pipeline, args=(job_id, params), daemon=True)
    thread.start()

    return PipelineResponse(
        job_id=job_id,
        status=PipelineStatus.PENDING,
        message="Pipeline job submitted. Poll /api/pipeline/status/{job_id} for progress.",
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Check the status of a pipeline job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        document_filename=job.get("document_filename"),
        project_name=job.get("project_name"),
    )


@router.get("/download/{job_id}")
def download_document(job_id: str):
    """Download the generated document for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != PipelineStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job is not completed. Status: {job['status']}")
    doc_path = job.get("document_path")
    if not doc_path or not os.path.exists(doc_path):
        raise HTTPException(status_code=404, detail="Document file not found.")
    return FileResponse(
        path=doc_path,
        filename=job.get("document_filename", "document.docx"),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
