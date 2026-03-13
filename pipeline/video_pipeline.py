# pipeline/video_pipeline.py

"""
Video Pipeline Orchestrator.
Silent screen recording (no audio) → Vision AI → PDD Document.

Flow:
1. Extract key frames via SSIM scene detection
2. OCR all frames (parallel)
3. Detect auth screens
4. Analyze transitions with vision AI (smart selection)
5. Synthesize detailed screen-by-screen steps (Section 2.4)
6. Infer logical high-level process steps (Section 2.2.2 & Flowchart)  <-- NEW
7. Generate document sections (parallel)
8. Generate flowchart
9. Annotate frames
10. Assemble PDD document
"""

import os
import time
import cv2
import concurrent.futures
from typing import Optional, Dict, List

from core.config import config
from core.gemini_client import gemini_client
from core.token_tracker import reset_tracker
from core.utils import detect_auth_screen, detect_operations_delta

from video.scene_detector import detect_scene_changes, get_video_info
from video.smart_sampler import select_key_frames, compute_target_frames
from video.ocr_engine import ocr_frame, OCR_AVAILABLE
from video.change_detector import detect_changes_between_frames
from video.frame_annotator import annotate_frame

from core.vectordb_client import upload_video, query_locate
from llm_tasks.meeting_compact import (
    generate_pdd_bundle_batch,
    generate_dot_from_transcript,
)

from pipeline.common import (
    save_persistent_document,
    save_dot_code,
    generate_flowchart,
    build_document,
    print_pipeline_header,
    print_pipeline_footer,
)


def _compute_max_vision_calls(num_frames: int) -> int:
    """Scale vision calls with frame count."""
    base = config.llm.min_vision_calls
    scaled = (num_frames // 10) * config.llm.vision_calls_per_10_frames
    target = max(base, scaled)
    return min(target, config.llm.absolute_max_vision_calls)


def _parallel_ocr(frame_paths: List[str], with_boxes: bool = False) -> Dict[str, Dict]:
    """Perform OCR on multiple frames in parallel."""
    results = {}
    if not OCR_AVAILABLE:
        print("    [OCR] Tesseract not available, returning empty results")
        return {fp: {"text": "", "boxes": [], "word_count": 0} for fp in frame_paths}

    print(f"    [OCR] Processing {len(frame_paths)} frames in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_path = {
            executor.submit(ocr_frame, fp, with_boxes): fp for fp in frame_paths
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as e:
                print(f"    [OCR] Error on {os.path.basename(path)}: {e}")
                results[path] = {"text": "", "boxes": [], "word_count": 0}

    non_empty = sum(1 for v in results.values() if v["text"].strip())
    print(f"    [OCR] {non_empty}/{len(frame_paths)} frames had readable text")
    return results


def _extract_micro_frames(
    video_path: str, scene_changes: List[Dict], output_dir: str, fps: float
) -> List[Dict]:
    """Extract additional frames around each scene change."""
    if not scene_changes:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    enhanced_frames = list(scene_changes)
    existing_timestamps = set(round(sc.get("timestamp", 0), 1) for sc in scene_changes)

    offsets = [-0.3, 0.2]
    micro_count = 0

    for i, scene in enumerate(scene_changes):
        timestamp = scene.get("timestamp", 0)
        for offset in offsets:
            t = timestamp + offset
            if t < 0.1 or t > duration - 0.1:
                continue
            if any(abs(t - et) < 0.15 for et in existing_timestamps):
                continue

            frame_idx = int(t * fps)
            if frame_idx < 0 or frame_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                minutes = int(t // 60)
                seconds = int(t % 60)
                millis = int((t % 1) * 100)
                filename = f"frame_micro_{i:03d}_off{int(offset * 10):+d}_{minutes}m{seconds:02d}s{millis:02d}.jpg"
                path = os.path.join(output_dir, filename)
                cv2.imwrite(path, frame)

                enhanced_frames.append(
                    {
                        "path": path,
                        "timestamp": t,
                        "frame_index": frame_idx,
                        "ssim_score": -1.0,
                        "is_micro_frame": True,
                    }
                )
                existing_timestamps.add(round(t, 1))
                micro_count += 1

    cap.release()
    enhanced_frames.sort(key=lambda x: x.get("timestamp", 0))

    if micro_count > 0:
        print(f"    [MicroFrames] Added {micro_count} micro-frames")
    return enhanced_frames


class VideoPipeline:
    """Pipeline for silent screen recordings."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or config.paths.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process(
        self,
        video_path: str,
        project_name: str,
        ssim_threshold: Optional[float] = None,
        max_frames: Optional[int] = None,
        annotate: Optional[bool] = None,
        enable_micro_frames: bool = True,
    ) -> Optional[str]:
        """Process a silent screen recording into a PDD document."""
        t0 = time.time()
        tracker = reset_tracker()
        gemini_client.set_tracker(tracker)

        print_pipeline_header(
            "Silent Screen Recording",
            video_path=video_path,
            project_name=project_name,
            extra_info={
                "OCR": "Available" if OCR_AVAILABLE else "Not available",
                "Workers": str(config.llm.max_workers),
            },
        )

        if not os.path.exists(video_path):
            print(f"Error: Video not found: {video_path}")
            return None

        # ── PHASE 1: Frame Extraction via VectorDB ──
        print(f"\n{'=' * 40}")
        print("PHASE 1/4: Uploading video & extracting frames...")
        print(f"{'=' * 40}")
        t = time.time()

        try:
            upload_res = upload_video(video_path)
            video_id = upload_res["video_id"]
            print(f"  VectorDB Upload Success! video_id: {video_id}")
            print(
                f"  Extracted {upload_res['frames_extracted']} frames in {time.time() - t:.1f}s"
            )
        except Exception as e:
            print(f"  Error uploading to VectorDB: {e}")
            return None

        frames_dir = os.path.join(
            os.getcwd(), "vectordb_service", "data", video_id, "frames"
        )
        image_paths = []
        if os.path.exists(frames_dir):
            import glob

            image_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

        if not image_paths:
            print("Error: No key frames extracted from VectorDB")
            return None

        # ── PHASE 2: OCR ──
        print(f"\n{'=' * 40}")
        print("PHASE 2/4: OCR...")
        print(f"{'=' * 40}")
        t = time.time()

        ocr_results = _parallel_ocr(image_paths, with_boxes=False)
        ocr_texts = []
        for p in image_paths:
            txt = ocr_results.get(p, {}).get("text", "").strip()
            if txt:
                ocr_texts.append(f"Frame {os.path.basename(p)} OCR:\n{txt}")
        combined_ocr = "\n\n".join(ocr_texts)

        print(f"  ✓ OCR complete ({time.time() - t:.0f}s)")

        # ── PHASE 3: Batch LLM Generation ──
        print(f"\n{'=' * 40}")
        print("PHASE 3/4: Batch LLM Generation...")
        print(f"{'=' * 40}")
        t = time.time()

        bundle = generate_pdd_bundle_batch(
            transcript=f"This is a silent screen recording. Here is the OCR text from the frames:\n{combined_ocr}",
            image_paths=image_paths,
            audio_path=None,
            project_name_hint=project_name,
        )

        if not project_name:
            project_name = bundle["project_name"]

        doc = bundle["document"]
        proc = bundle["process"]
        reqs = bundle["requirements"]

        process_steps = proc.get("process_steps", [])
        detailed_steps = proc.get("detailed_steps", [])

        # Flowchart Generation
        dot_code = generate_dot_from_transcript(
            combined_ocr, project_name, process_steps
        )
        if dot_code:
            save_dot_code(dot_code, project_name, self.output_dir)
        fc_path = generate_flowchart(dot_code, self.output_dir, project_name)

        print(f"  ✓ Phase 3 complete ({time.time() - t:.0f}s)")

        # ── PHASE 4: Document Assembly & VectorDB Locate ──
        print(f"\n{'=' * 40}")
        print("PHASE 4/4: Locating UI Elements & Assembly...")
        print(f"{'=' * 40}")
        t = time.time()

        detailed_dicts = []
        for i, s in enumerate(detailed_steps):
            if isinstance(s, dict):
                detailed_dicts.append(
                    {
                        "number": f"2.4.{i + 1}",
                        "description": s.get("action", ""),
                        "ui_target": s.get("ui_target", ""),
                    }
                )
            else:
                detailed_dicts.append(
                    {"number": f"2.4.{i + 1}", "description": s, "ui_target": s}
                )

        annotated_frames = {}
        for i, step in enumerate(detailed_dicts):
            try:
                query_text = step.get("ui_target") or step["description"]
                res = query_locate(query=query_text, video_id=video_id, top_k=5)
                url = res.get("annotated_image_url", "")
                if url:
                    filename = os.path.basename(url)
                    local_annotated_path = os.path.join(frames_dir, filename)
                    if os.path.exists(local_annotated_path):
                        step["frame_after_path"] = local_annotated_path
                        num_key = int(str(step["number"]).split(".")[-1])
                        annotated_frames[num_key] = local_annotated_path
            except Exception as e:
                print(f"    Failed to locate frame for step {step['number']}: {e}")

        process_steps_dicts = (
            [{"number": i + 1, "description": s} for i, s in enumerate(process_steps)]
            if process_steps
            else []
        )

        doc_path = build_document(
            project_name=project_name,
            output_dir=self.output_dir,
            purpose=doc.get("purpose", ""),
            overview=doc.get("overview", ""),
            justification=doc.get("justification", ""),
            as_is=doc.get("as_is", ""),
            to_be=doc.get("to_be", ""),
            process_steps=process_steps_dicts,
            input_requirements=reqs.get("input_requirements", []),
            detailed_steps=detailed_dicts,
            interface_requirements=reqs.get("interface_requirements", []),
            exception_handling=reqs.get("exception_handling", []),
            flowchart_path=fc_path,
            annotated_frames=annotated_frames,
        )

        persistent = save_persistent_document(doc_path, project_name)
        tracker.print_report()
        tracker.save_csv(project_name)

        stats = {
            "Process Steps": len(process_steps_dicts),
            "Detailed Steps": len(detailed_dicts),
            "Frames": len(image_paths),
            "Annotated Frames": len(annotated_frames),
        }
        print_pipeline_footer(persistent, project_name, stats, time.time() - t0)
        return doc_path
