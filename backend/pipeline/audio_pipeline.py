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
from typing import Optional

from core.config import config
from core.gemini_client import gemini_client
from core.token_tracker import reset_tracker

from audio.video_to_audio import convert_video_to_audio
from audio.transcriber import transcribe_audio, read_transcript

from core.vectordb_client import upload_video, query_locate, download_all_frames, download_frame

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


class AudioPipeline:
    """Pipeline for meeting recordings with audio — consolidated LLM calls."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.paths.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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

        # Download frames from VectorDB service
        frames_dir = os.path.join(self.output_dir, "frames")
        image_paths = download_all_frames(video_id, frames_dir)

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
                        "screenshot_query": s.get("screenshot_query", ""),
                    }
                )
            else:
                detailed_dicts.append(
                    {"number": f"2.4.{i + 1}", "description": s, "ui_target": s, "screenshot_query": s}
                )

        print("  Querying VectorDB to map detailed steps to annotated frames...")
        frames_matched = 0
        for i, step in enumerate(detailed_dicts):
            try:
                query_text = step.get("screenshot_query") or f"UI Element like {step.get('ui_target')} describing {step['description']}"
                res = query_locate(query=query_text, video_id=video_id, top_k=5)
                url = res.get("annotated_image_url", "")
                if url:
                    filename = os.path.basename(url)
                    local_path = download_frame(video_id, filename, frames_dir)
                    if local_path:
                        step["frame_after_path"] = local_path
                        frames_matched += 1
                    else:
                        print(f"    [Warn] Could not download annotated frame: {filename}")
            except Exception as e:
                print(f"    Failed to locate frame for step {step['number']}: {e}")

        print("\n[5/5] Generating document...")

        process_steps_dicts = None
        if process_steps:
            process_steps_dicts = [
                {"number": i + 1, "description": s} for i, s in enumerate(process_steps)
            ]

        if frames_matched:
            print(f"  {frames_matched} frames will be embedded in document")

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
                "Frames embedded": frames_matched,
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
