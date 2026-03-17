"""MongoDB GridFS storage for extracted video frames and annotated images."""

import io
import logging
import os
from typing import Optional

from bson import ObjectId
from PIL import Image
from pymongo import MongoClient
from gridfs import GridFS

logger = logging.getLogger(__name__)


class MongoFrameStore:
    """Stores and retrieves video frames in MongoDB using GridFS."""

    def __init__(self, mongo_uri: str, db_name: str = "pdd_vectordb") -> None:
        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]
        self._fs = GridFS(self._db)
        # Collection for frame metadata (fast lookups by video_id + filename)
        self._meta = self._db["frame_metadata"]
        self._meta.create_index([("video_id", 1), ("filename", 1)], unique=True)
        self._meta.create_index("video_id")
        logger.info("MongoFrameStore connected to %s / %s", mongo_uri, db_name)

    def save_frame(
        self,
        video_id: str,
        filename: str,
        image_bytes: bytes,
        content_type: str = "image/png",
        metadata: Optional[dict] = None,
    ) -> str:
        """Save a frame image to GridFS.

        Returns:
            The GridFS file_id as a hex string.
        """
        extra = metadata or {}
        gridfs_id = self._fs.put(
            image_bytes,
            filename=filename,
            content_type=content_type,
            video_id=video_id,
            **{k: v for k, v in extra.items() if k not in ("video_id", "filename")},
        )
        self._meta.update_one(
            {"video_id": video_id, "filename": filename},
            {
                "$set": {
                    "gridfs_id": gridfs_id,
                    "content_type": content_type,
                    **(extra),
                }
            },
            upsert=True,
        )
        return str(gridfs_id)

    def save_frame_from_path(
        self,
        video_id: str,
        filepath: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Read an image file from disk and store it in GridFS."""
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        ct = "image/png" if ext == ".png" else "image/jpeg"
        with open(filepath, "rb") as f:
            data = f.read()
        return self.save_frame(video_id, filename, data, content_type=ct, metadata=metadata)

    def get_frame_bytes(self, video_id: str, filename: str) -> Optional[bytes]:
        """Retrieve frame image bytes by video_id and filename."""
        doc = self._meta.find_one({"video_id": video_id, "filename": filename})
        if not doc:
            return None
        gridfs_id = doc.get("gridfs_id")
        if not gridfs_id:
            return None
        try:
            grid_out = self._fs.get(gridfs_id)
            return grid_out.read()
        except Exception:
            logger.exception("Failed to read frame %s/%s from GridFS", video_id, filename)
            return None

    def get_frame_as_pil(self, video_id: str, filename: str) -> Optional[Image.Image]:
        """Retrieve frame as a PIL Image."""
        data = self.get_frame_bytes(video_id, filename)
        if data is None:
            return None
        return Image.open(io.BytesIO(data))

    def write_frame_to_tempfile(self, video_id: str, filename: str, temp_dir: str) -> Optional[str]:
        """Write a frame from GridFS to a temporary file and return its path.

        This is needed by components that require a filesystem path (e.g., OmniParser, CLIP).
        """
        data = self.get_frame_bytes(video_id, filename)
        if data is None:
            return None
        os.makedirs(temp_dir, exist_ok=True)
        path = os.path.join(temp_dir, filename)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def list_frames(self, video_id: str) -> list[dict]:
        """List all frame metadata documents for a video."""
        return list(
            self._meta.find(
                {"video_id": video_id},
                {"_id": 0, "gridfs_id": 0},
            )
        )

    def delete_video_frames(self, video_id: str) -> int:
        """Delete all frames for a given video from GridFS and metadata."""
        docs = list(self._meta.find({"video_id": video_id}))
        count = 0
        for doc in docs:
            gid = doc.get("gridfs_id")
            if gid:
                try:
                    self._fs.delete(gid)
                    count += 1
                except Exception:
                    pass
        self._meta.delete_many({"video_id": video_id})
        return count
