"""Route to serve frame images stored in MongoDB GridFS."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from vectordb_service.api.dependencies import get_mongo_frame_store
from infrastructure.mongo_frame_store import MongoFrameStore

router = APIRouter(prefix="/api/frames", tags=["frames"])


@router.get("/image/{video_id}/{filename}")
def get_frame_image(
    video_id: str,
    filename: str,
    frame_store: MongoFrameStore = Depends(get_mongo_frame_store),
):
    """Retrieve a frame image from MongoDB GridFS."""
    data = frame_store.get_frame_bytes(video_id, filename)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Frame not found: {video_id}/{filename}")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    content_type = "image/png" if ext == "png" else "image/jpeg"
    return Response(content=data, media_type=content_type)
