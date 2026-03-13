# pipeline/audio_pipeline.py

"""
Audio Pipeline Orchestrator.
Meeting recording (with audio) → Transcript → LLM → PDD Document.

Consolidated flow (3 LLM calls):
1. Extract audio / use provided transcript
2. LLM Call 1: Document sections
3. LLM Call 2: Process steps & requirements
4. LLM Call 3: Step refinement — decompose into granular sub-steps
5. LLM Call 4: DOT flowchart
6. Extract frames & attach to steps
7. Assemble PDD document
"""

import os
import time
import re
from typing import Optional, List, Tuple, Dict

from core.config import config
from core.gemini_client import gemini_client
from core.token_tracker import reset_tracker
from core.utils import build_entity_hint, redact_pii_from_image

from audio.video_to_audio import convert_video_to_audio
from audio.transcriber import transcribe_audio, read_transcript

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
from document.pdd_generator import PDDGenerator


class AudioPipeline:
    """Pipeline for meeting recordings with audio — consolidated LLM calls."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.paths.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_evenly_spaced_frames(
        self, video_path: str, frames_dir: str, num_frames: int
    ) -> List[Tuple[str, float]]:
        """Extract evenly spaced frames across the entire video."""
        import cv2

        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("    [Frames] Cannot open video")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames_count / fps if fps > 0 else 0

        if duration <= 0:
            cap.release()
            print("    [Frames] Cannot determine video duration")
            return []

        print(f"    [Frames] Video: {duration:.0f}s, extracting {num_frames} frames...")

        start_t = duration * 0.02
        end_t = duration * 0.98
        interval = (end_t - start_t) / (num_frames + 1)

        frames = []
        for i in range(num_frames):
            timestamp = start_t + interval * (i + 1)
            frame_idx = int(timestamp * fps)

            if frame_idx >= total_frames_count:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                filename = f"frame_{i:03d}_{minutes}m{seconds:02d}s.jpg"
                frame_path = os.path.join(frames_dir, filename)
                cv2.imwrite(frame_path, frame)

                redact_pii_from_image(frame_path)
                frames.append((frame_path, timestamp))

        cap.release()
        print(f"    [Frames] Extracted {len(frames)} frames")
        return frames

    def _extract_keyword_frames(
        self,
        video_path: str,
        transcript_path: str,
        frames_dir: str,
        max_frames: int = 30,
    ) -> List[Tuple[str, float, str]]:
        """Extract frames at transcript action keyword timestamps."""
        import cv2

        os.makedirs(frames_dir, exist_ok=True)

        lines = []
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.match(
                        r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)", line.strip()
                    )
                    if m:
                        lines.append(
                            {"timestamp": float(m.group(1)), "text": m.group(3).strip()}
                        )
        except Exception as e:
            print(f"    [Frames] Error reading transcript: {e}")
            return []

        if not lines:
            return []

        from core.config import ACTION_KEYWORDS

        all_kw = set()
        for kl in ACTION_KEYWORDS.values():
            for kw in kl:
                all_kw.add(kw.lower())

        action_lines = []
        for tl in lines:
            if any(kw in tl["text"].lower() for kw in all_kw):
                action_lines.append(tl)

        if not action_lines:
            return []

        deduped = [action_lines[0]]
        for al in action_lines[1:]:
            if al["timestamp"] - deduped[-1]["timestamp"] > 3.0:
                deduped.append(al)

        if len(deduped) > max_frames:
            step = len(deduped) // max_frames
            deduped = deduped[::step][:max_frames]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        for i, al in enumerate(deduped):
            ts = al["timestamp"]
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()

            if ret and frame is not None:
                minutes = int(ts // 60)
                seconds = int(ts % 60)
                filename = f"frame_kw_{i:03d}_{minutes}m{seconds:02d}s.jpg"
                frame_path = os.path.join(frames_dir, filename)
                cv2.imwrite(frame_path, frame)

                redact_pii_from_image(frame_path)
                frames.append((frame_path, ts, al["text"]))

        cap.release()
        print(
            f"    [Frames] Extracted {len(frames)} keyword frames from {len(deduped)} timestamps"
        )
        return frames

    def _assign_frames_to_steps(
        self,
        frames: List[Tuple[str, float]],
        detailed_dicts: List[Dict],
    ) -> Dict[str, str]:
        """Assign frames to detailed steps by distributing evenly."""
        if not frames or not detailed_dicts:
            return {}

        num_steps = len(detailed_dicts)
        num_frames = len(frames)

        sorted_frames = sorted(frames, key=lambda x: x[1])

        assigned = {}

        if num_frames >= num_steps:
            for i, step in enumerate(detailed_dicts):
                frame_idx = int(i * num_frames / num_steps)
                frame_idx = min(frame_idx, num_frames - 1)
                step_num = step.get("number", f"2.4.{i + 1}")
                assigned[str(step_num)] = sorted_frames[frame_idx][0]
        else:
            interval = max(1, num_steps // num_frames)
            frame_idx = 0
            for i, step in enumerate(detailed_dicts):
                if frame_idx < num_frames and (i % interval == 0 or i == 0):
                    step_num = step.get("number", f"2.4.{i + 1}")
                    assigned[str(step_num)] = sorted_frames[frame_idx][0]
                    frame_idx += 1

        print(f"    [Frames] Assigned {len(assigned)} frames to {num_steps} steps")
        return assigned

    def process(
        self,
        video_path: str,
        project_name: str = None,
        whisper_model: str = None,
        transcript_path: str = None,
    ) -> Optional[str]:
        """Process a meeting recording into a PDD document."""
        t0 = time.time()
        tracker = reset_tracker()
        gemini_client.set_tracker(tracker)

        print_pipeline_header(
            "Meeting Recording (Audio) — Consolidated",
            video_path=video_path,
            project_name=project_name or "(auto-detect)",
            extra_info={
                "Mode": "Consolidated (3-4 LLM calls)",
            },
        )

        if not os.path.exists(video_path):
            print(f"Error: {video_path} not found")
            return None

        print("\n[1/5] Extracting audio and transcribing...")
        if transcript_path and os.path.exists(transcript_path):
            print(f"  Using provided transcript: {transcript_path}")
        else:
            audio = convert_video_to_audio(video_path, self.output_dir)
            if not audio:
                return None
            transcript_path = transcribe_audio(
                audio,
                self.output_dir,
                model_name=whisper_model or config.whisper.model_name,
            )
            if not transcript_path:
                return None

        transcript = read_transcript(transcript_path)
        if not transcript:
            return None
        print(f"  Transcript: {len(transcript):,} chars")

        print(
            "\n[2/5] Uploading video to VectorDB for scene extraction and indexing..."
        )
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

        print(f"\n[3/5] Batch LLM extraction (1 call)...")
        t = time.time()

        audio_for_gemini = (
            transcript_path.replace(".txt", ".mp3") if transcript_path else None
        )
        if not (audio_for_gemini and os.path.exists(audio_for_gemini)):
            audio_for_gemini = None

        bundle = generate_pdd_bundle_batch(
            transcript=transcript,
            image_paths=image_paths,
            audio_path=audio_for_gemini,
            project_name_hint=project_name,
        )

        if not project_name:
            project_name = bundle["project_name"]

        doc = bundle["document"]
        proc = bundle["process"]
        reqs = bundle["requirements"]
        entities = bundle["entities"]

        process_steps = proc.get("process_steps", [])
        detailed_steps = proc.get("detailed_steps", [])

        print(f"  Project: {project_name}")
        print(f"  Purpose: {len(doc.get('purpose', ''))} chars")
        print(f"  Overview: {len(doc.get('overview', ''))} chars")
        print(f"  As-Is: {len(doc.get('as_is', ''))} chars")
        print(f"  To-Be: {len(doc.get('to_be', ''))} chars")
        print(f"  Process steps: {len(process_steps)}")
        print(f"  Detailed steps: {len(detailed_steps)}")
        print(f"  Inputs: {len(reqs.get('input_requirements', []))}")
        print(f"  Interfaces: {len(reqs.get('interface_requirements', []))}")
        print(f"  Exceptions: {len(reqs.get('exception_handling', []))}")
        print(f"  ({time.time() - t:.0f}s)")

        print("\n[4/5] Flowchart & screenshots...")

        t = time.time()
        dot_code = generate_dot_from_transcript(transcript, project_name, process_steps)
        if dot_code:
            save_dot_code(dot_code, project_name, self.output_dir)
        fc_path = generate_flowchart(dot_code, self.output_dir, project_name)
        print(f"  Flowchart ({time.time() - t:.0f}s)")

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

        print("  Querying VectorDB to map detailed steps to annotated frames...")
        annotated_frames = {}
        for i, step in enumerate(detailed_dicts):
            try:
                query_text = step.get("ui_target") or step["description"]
                res = query_locate(query=query_text, video_id=video_id, top_k=5)
                # annotated_image_url example: /static/vid/frames/annotated.jpg
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

        print("\n[5/5] Generating document...")

        process_steps_dicts = None
        if process_steps:
            process_steps_dicts = [
                {"number": i + 1, "description": s} for i, s in enumerate(process_steps)
            ]

        if annotated_frames:
            print(f"  {len(annotated_frames)} frames will be embedded in document")

        doc_path = build_document(
            project_name=project_name,
            output_dir=self.output_dir,
            purpose=doc.get("purpose", ""),
            overview=doc.get("overview", ""),
            justification=doc.get("justification", ""),
            as_is=doc.get("as_is", ""),
            to_be=doc.get("to_be", ""),
            process_steps=process_steps_dicts or [],
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

        total = time.time() - t0
        print_pipeline_footer(
            persistent,
            project_name,
            {
                "Steps": len(process_steps),
                "Detailed": len(detailed_steps),
                "Frames embedded": len(annotated_frames),
                "LLM Calls": len(tracker.calls),
            },
            total,
        )
        return doc_path

    @staticmethod
    def _ocr_available() -> bool:
        try:
            from video.ocr_engine import OCR_AVAILABLE

            return OCR_AVAILABLE
        except ImportError:
            return False
