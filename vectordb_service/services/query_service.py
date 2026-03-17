"""Handles text-based frame querying with Gemini frame selection, OmniParser analysis, and bounding box annotation."""

import io
import logging
import os
import tempfile

import replicate

from config import Settings
from infrastructure.bbox_drawer import draw_highlight, draw_highlight_to_bytes
from infrastructure.embedding_service import EmbeddingService
from infrastructure.gemini_frame_selector import locate_bounding_box, select_best_frame
from infrastructure.mongo_frame_store import MongoFrameStore
from infrastructure.omniparser_client import parse_frame
from infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryService:
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
        self._replicate = replicate.Client(api_token=settings.replicate_api_token)

    def query_frames(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> dict:
        # CLIP text embedding (768-d) — directly comparable to image embeddings
        text_emb = self._embedding.get_text_embedding(query_text)
        query_vector = text_emb.flatten().tolist()

        metadata_filter = {"video_id": {"$eq": video_id}} if video_id else None
        results = self._vector_store.query(query_vector, top_k=top_k, filter=metadata_filter)

        matches: list[dict] = []
        for m in results.get("matches", []):
            meta = m.get("metadata", {})
            vid = meta.get("video_id", "")
            filename = meta.get("filename", "")
            # Frame images are now served via /api/frames/image/{video_id}/{filename}
            image_url = f"/api/frames/image/{vid}/{filename}" if vid and filename else ""

            matches.append(
                {
                    "vector_id": m["id"],
                    "score": m["score"],
                    "metadata": meta,
                    "image_url": image_url,
                }
            )

        return {"query": query_text, "results": matches}

    def query_and_locate(self, query_text: str, top_k: int = 5, video_id: str | None = None) -> dict:
        """Full pipeline: vector search → Gemini frame selection → OmniParser → Gemini bbox → draw bbox."""

        # Step 1: Vector similarity search
        search_result = self.query_frames(query_text, top_k=top_k, video_id=video_id)
        matches = search_result["results"]

        if not matches:
            return {"query": query_text, "error": "No matching frames found"}

        # Write matched frames from MongoDB to temp files (needed by Gemini/OmniParser)
        with tempfile.TemporaryDirectory(prefix="pdd_query_") as tmp_dir:
            frame_paths: list[str] = []
            for m in matches:
                meta = m["metadata"]
                vid = meta.get("video_id", "")
                filename = meta.get("filename", "")
                tmp_path = self._frame_store.write_frame_to_tempfile(vid, filename, tmp_dir)
                if tmp_path:
                    frame_paths.append(tmp_path)
                else:
                    frame_paths.append("")

            valid_paths = [p for p in frame_paths if p]
            if not valid_paths:
                return {"query": query_text, "error": "No frame images could be retrieved from MongoDB"}

            # Step 2: Gemini picks the best frame
            logger.info("Asking Gemini to select best frame for: %s", query_text)
            selection = select_best_frame(
                api_key=self._settings.gemini_api_key,
                query=query_text,
                frame_paths=[p for p in frame_paths if p],
                model=self._settings.gemini_model,
            )
            selected_idx = selection.get("selected_frame_index", 0)
            selected_idx = max(0, min(selected_idx, len(valid_paths) - 1))
            selected_path = valid_paths[selected_idx]
            # Map back to the original match
            valid_match_indices = [i for i, p in enumerate(frame_paths) if p]
            selected_match = matches[valid_match_indices[selected_idx]]

            # Step 3: OmniParser analyses the selected frame
            logger.info("Running OmniParser on %s", selected_path)
            omniparser_output = parse_frame(
                image_path=selected_path,
                model_version=self._settings.omniparser_model,
                client=self._replicate,
            )

            # Extract elements with bounding boxes for Gemini
            elements = self._extract_elements(omniparser_output)

            # Step 4: Gemini locates the target region
            logger.info("Asking Gemini to locate region for: %s", query_text)
            try:
                bbox_result = locate_bounding_box(
                    api_key=self._settings.gemini_api_key,
                    query=query_text,
                    frame_path=selected_path,
                    omniparser_elements=elements,
                    model=self._settings.gemini_model,
                )
            except Exception:
                logger.exception("Gemini locate_bounding_box failed")
                bbox_result = {"element_text": "", "highlight_type": "none", "region": None, "reason": "Gemini API error"}

            highlight_type = bbox_result.get("highlight_type", "none")
            if highlight_type not in ("circle", "square", "none"):
                highlight_type = "none"
            region = bbox_result.get("region") or [0, 0, 0, 0]

            # Step 5: Draw highlight and store annotated image in MongoDB
            vid = selected_match["metadata"].get("video_id", "")
            annotated_bytes = draw_highlight_to_bytes(
                image_path=selected_path,
                region=region,
                highlight_type=highlight_type,
            )
            base = os.path.splitext(selected_match["metadata"].get("filename", "frame"))[0]
            annotated_filename = f"{base}_annotated.png"

            self._frame_store.save_frame(
                video_id=vid,
                filename=annotated_filename,
                image_bytes=annotated_bytes,
                content_type="image/png",
                metadata={"type": "annotated", "query": query_text},
            )
            annotated_url = f"/api/frames/image/{vid}/{annotated_filename}"

        return {
            "query": query_text,
            "selected_frame": selected_match,
            "selection_reason": selection.get("reason", ""),
            "omniparser_elements_count": len(elements),
            "located_element": bbox_result.get("element_text", ""),
            "highlight_type": highlight_type,
            "region": region,
            "region_reason": bbox_result.get("reason", ""),
            "annotated_image_url": annotated_url,
        }

    @staticmethod
    def _extract_elements(omniparser_output: dict) -> list[dict]:
        """Parse OmniParser raw output into a list of elements with bbox."""
        raw_elements = omniparser_output.get("elements", "")
        elements: list[dict] = []
        for line in raw_elements.split("\n"):
            if "{" not in line:
                continue
            try:
                import ast
                obj_str = line.split(":", 1)[1].strip()
                obj = ast.literal_eval(obj_str)
                elements.append({
                    "type": obj.get("type", ""),
                    "content": obj.get("content", ""),
                    "bbox": obj.get("bbox", []),
                })
            except Exception:
                continue
        return elements
