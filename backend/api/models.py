"""
Pydantic models for Backend API requests and responses.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class InputMode(str, Enum):
    MEETING = "meeting"
    SILENT = "silent"


class DocumentType(str, Enum):
    PDD = "PDD"
    BRD = "BRD"


class PipelineRequest(BaseModel):
    """Request to start a pipeline processing job."""
    input_mode: InputMode
    project_name: Optional[str] = None
    document_type: DocumentType = DocumentType.PDD

    # Audio pipeline settings
    whisper_model: str = "base"
    # If transcript text is provided directly (instead of auto-transcribing)
    transcript_text: Optional[str] = None

    # Video pipeline settings
    ssim_threshold: float = 0.92
    max_frames: int = 60
    enable_micro_frames: bool = True
    annotate: bool = True

    # Privacy
    pii_redaction: bool = True


class PipelineStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineResponse(BaseModel):
    """Response after submitting a pipeline job."""
    job_id: str
    status: PipelineStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: PipelineStatus
    message: str
    document_filename: Optional[str] = None
    project_name: Optional[str] = None
    stats: Optional[dict] = None


class TokenUsageSummary(BaseModel):
    calls: int = 0
    total_tokens_est: int = 0
    total_tokens_actual: int = 0
    duration_seconds: float = 0.0


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    gemini_available: bool
    gemini_model: Optional[str] = None
