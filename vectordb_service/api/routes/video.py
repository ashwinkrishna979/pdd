from fastapi import APIRouter, Depends, HTTPException, UploadFile

from vectordb_service.api.dependencies import get_video_service
from domain.models import UploadResponse
from services.video_service import VideoService

router = APIRouter(prefix="/api/videos", tags=["videos"])

_ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@router.post("/upload", response_model=UploadResponse)
def upload_video(
    file: UploadFile,
    video_service: VideoService = Depends(get_video_service),
):
    """Upload a video file. Scenes are detected, frames extracted, parsed with
    OmniParser, embedded via CLIP, and indexed in Pinecone."""

    filename = file.filename or "upload.mp4"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {_ALLOWED_EXTENSIONS}",
        )

    video_bytes = file.file.read()
    result = video_service.process_video(video_bytes, filename)

    return UploadResponse(
        video_id=result["video_id"],
        scenes_detected=result["scenes_detected"],
        frames_extracted=result["frames_extracted"],
        frames_indexed=result["frames_indexed"],
        message="Video processed and indexed successfully.",
    )
