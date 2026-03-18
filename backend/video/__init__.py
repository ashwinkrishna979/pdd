# video/__init__.py

"""
Video processing modules for both pipelines.
Scene detection, frame sampling, OCR, change detection,
frame extraction, frame matching, and annotation.
"""

from video.scene_detector import detect_scene_changes, get_video_info
from video.smart_sampler import select_key_frames, compute_target_frames
from video.ocr_engine import ocr_frame, ocr_batch, OCR_AVAILABLE
from video.change_detector import detect_changes_between_frames
from video.frame_annotator import annotate_frame
from video.frame_extractor import (
    extract_frame, get_video_duration,
    extract_frames_with_transcripts,
    extract_evenly_spaced_frames
)
from video.frame_matcher import match_pipeline, build_candidates