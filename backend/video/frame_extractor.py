# audio/frame_extractor.py

"""
Video frame extraction using OpenCV.
Extracts frames at specific timestamps or evenly spaced.
Used by the audio pipeline to capture screenshots matching transcript actions.
"""

import os
import re
from typing import List, Tuple, Dict, Optional
import cv2

from core.config import ACTION_KEYWORDS


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return frames / fps


def extract_timestamps_from_transcript(
    transcript_path: str,
    keywords: Dict[str, List[str]] = None
) -> Tuple[List[float], Dict[float, str]]:
    """Extract timestamps where action keywords are mentioned."""
    keywords = keywords or ACTION_KEYWORDS

    if not transcript_path or not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found: {transcript_path}")
        return [], {}

    timestamps = []
    transcript_dict = {}

    all_keywords = set(
        word.lower()
        for keyword_list in keywords.values()
        for word in keyword_list
    )

    pattern = re.compile(r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)")

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                start_time = float(match.group(1))
                text = match.group(3).strip()
                transcript_dict[start_time] = text

                text_lower = text.lower()
                if any(kw in text_lower for kw in all_keywords):
                    timestamps.append(start_time)

    print(f"Found {len(timestamps)} action timestamps")
    return timestamps, transcript_dict


def extract_frame(
    video_path: str, timestamp: float, output_dir: str
) -> Optional[str]:
    """Extract a single frame from video at specific timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    frame_number = int(timestamp * fps)
    frame_path = os.path.join(
        output_dir, f"frame_{frame_number}_{timestamp:.2f}.jpg"
    )
    cv2.imwrite(frame_path, frame)
    return frame_path


def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    output_dir: str
) -> List[str]:
    """Extract multiple frames at given timestamps."""
    frame_paths = []
    for timestamp in timestamps:
        frame_path = extract_frame(video_path, timestamp, output_dir)
        if frame_path:
            frame_paths.append(frame_path)
    print(f"Extracted {len(frame_paths)} frames")
    return frame_paths


def extract_evenly_spaced_frames(
    video_path: str,
    output_dir: str,
    num_frames: int = 15
) -> List[Tuple[str, str]]:
    """
    Extract frames evenly spaced across the video.
    Fallback method when timestamp-based extraction fails.

    Returns list of (frame_path, timestamp_description) tuples.
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        print("Cannot determine video duration")
        return []

    print(f"  Video duration: {duration:.0f}s, extracting {num_frames} evenly spaced frames")
    os.makedirs(output_dir, exist_ok=True)

    start = duration * 0.05
    end = duration * 0.95
    interval = (end - start) / (num_frames + 1)

    frame_pairs = []
    for i in range(num_frames):
        timestamp = start + interval * (i + 1)
        frame_path = extract_frame(video_path, timestamp, output_dir)
        if frame_path:
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            desc = f"Process step at {minutes}:{seconds:02d}"
            frame_pairs.append((frame_path, desc))

    print(f"  Extracted {len(frame_pairs)} evenly spaced frames")
    return frame_pairs


def extract_frames_with_transcripts(
    video_path: str,
    transcript_path: str,
    output_dir: str,
    keywords: Dict[str, List[str]] = None
) -> List[Tuple[str, str]]:
    """
    Extract frames at action timestamps with their transcript text.
    Falls back to evenly-spaced extraction if too few keyword matches.
    """
    timestamps, transcript_dict = extract_timestamps_from_transcript(
        transcript_path, keywords
    )

    frame_transcript_pairs = []

    if timestamps:
        # Deduplicate close timestamps
        deduped = [timestamps[0]]
        for ts in timestamps[1:]:
            if ts - deduped[-1] > 5.0:
                deduped.append(ts)
        timestamps = deduped

        # Limit to 15
        if len(timestamps) > 15:
            step = len(timestamps) // 15
            timestamps = timestamps[::step][:15]

        for timestamp in timestamps:
            frame_path = extract_frame(video_path, timestamp, output_dir)
            if frame_path:
                text = transcript_dict.get(timestamp, "Process step")
                frame_transcript_pairs.append((frame_path, text))

    # Fallback if too few
    if len(frame_transcript_pairs) < 5:
        print("  Too few keyword frames, adding evenly spaced frames...")
        evenly = extract_evenly_spaced_frames(
            video_path, output_dir,
            num_frames=15 - len(frame_transcript_pairs)
        )
        frame_transcript_pairs.extend(evenly)

    return frame_transcript_pairs