# audio/video_to_audio.py

"""
Video to Audio conversion using FFmpeg.
Extracts audio track from video files.
"""

import os
import subprocess
from typing import Optional

from core.config import config


def convert_video_to_audio(
    video_path: str,
    output_dir: str = None,
    ffmpeg_path: str = None
) -> Optional[str]:
    """
    Convert video file to MP3 audio using FFmpeg.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the audio file.
        ffmpeg_path: Path to FFmpeg executable (optional).

    Returns:
        Path to the extracted audio file, or None if failed.
    """
    output_dir = output_dir or config.paths.output_dir
    ffmpeg_path = ffmpeg_path or config.paths.ffmpeg_path

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}.mp3")

    if os.path.exists(audio_path):
        print(f"Audio file already exists: {audio_path}")
        return audio_path

    ffmpeg_cmd = [
        ffmpeg_path,
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "192k",
        "-ar", "44100",
        "-y",
        audio_path
    ]

    try:
        print(f"Converting video to audio: {video_path}")
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Audio saved to: {audio_path}")
        return audio_path

    except FileNotFoundError:
        print(f"Error: FFmpeg not found at '{ffmpeg_path}'")
        print("Please install FFmpeg or specify the correct path.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return None