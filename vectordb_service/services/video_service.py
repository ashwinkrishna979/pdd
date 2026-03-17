"""Orchestrates the full video-processing pipeline: scenes → frames → MongoDB + embeddings → Pinecone."""

import logging
import os
import tempfile
import uuid

from config import Settings
from infrastructure.embedding_service import EmbeddingService
from infrastructure.frame_extractor import extract_scene_frames
from infrastructure.mongo_frame_store import MongoFrameStore
from infrastructure.scene_detector import detect_scenes
from infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        frame_store: MongoFrameStore,
    ) -> None:
        self._settings = settings
        self._embedding = embedding_service
        self._vector_store = vector_store
        self._frame_store = frame_store

    # -- public ---------------------------------------------------------

    def process_video(self, video_bytes: bytes, original_filename: str) -> dict:
        video_id = uuid.uuid4().hex[:12]

        # Use a temp directory for intermediate files (video + extracted frames)
        with tempfile.TemporaryDirectory(prefix=f"pdd_{video_id}_") as tmp_dir:
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            # Persist uploaded file temporarily
            video_path = os.path.join(tmp_dir, original_filename)
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            # 1. Scene detection
            logger.info("Detecting scenes for video %s", video_id)
            scenes = detect_scenes(video_path, self._settings.scene_threshold)
            logger.info("Scenes detected: %d", len(scenes))

            # 2. Frame extraction to temp dir
            frames = extract_scene_frames(video_path, scenes, frames_dir)
            logger.info("Frames extracted: %d", len(frames))

            # 3. Store frames in MongoDB and generate embeddings
            vectors_to_upsert: list[tuple] = []

            for frame_info in frames:
                filepath = frame_info["filepath"]
                filename = frame_info["filename"]
                try:
                    # Store frame in MongoDB GridFS
                    self._frame_store.save_frame_from_path(
                        video_id=video_id,
                        filepath=filepath,
                        metadata={
                            "scene_id": frame_info["scene_id"],
                            "label": frame_info["label"],
                            "frame_no": frame_info["frame_no"],
                        },
                    )

                    # Generate CLIP embedding
                    image_emb = self._embedding.get_image_embedding(filepath)
                    embedding = image_emb.flatten().tolist()

                    metadata = {
                        "video_id": video_id,
                        "filename": filename,
                        "scene_id": frame_info["scene_id"],
                        "label": frame_info["label"],
                        "frame_no": frame_info["frame_no"],
                    }

                    vector_id = f"{video_id}_scene_{frame_info['scene_id']}_{frame_info['label']}"
                    vectors_to_upsert.append((vector_id, embedding, metadata))
                except Exception:
                    logger.exception(
                        "Processing failed for scene_%d_%s, skipping",
                        frame_info["scene_id"],
                        frame_info["label"],
                    )

            # 4. Upsert to Pinecone
            if vectors_to_upsert:
                self._vector_store.upsert(vectors_to_upsert)
                logger.info("Upserted %d vectors", len(vectors_to_upsert))

        # Temp dir (including video file and frame files) is auto-cleaned

        return {
            "video_id": video_id,
            "scenes_detected": len(scenes),
            "frames_extracted": len(frames),
            "frames_indexed": len(vectors_to_upsert),
        }

    def delete_video(self, video_id: str) -> dict:
        """Delete all MongoDB frames and Pinecone vectors for a video."""
        frames_deleted = self._frame_store.delete_video_frames(video_id)
        try:
            self._vector_store.delete_by_video_id(video_id)
        except Exception:
            logger.exception("Pinecone delete failed for video %s", video_id)
        logger.info("Deleted video %s: %d frames removed from MongoDB", video_id, frames_deleted)
        return {"video_id": video_id, "frames_deleted": frames_deleted}
