# audio/__init__.py

"""
Audio processing modules for the audio pipeline.
Video-to-audio extraction, Whisper transcription, frame extraction, frame matching.
"""

from audio.video_to_audio import convert_video_to_audio
from audio.transcriber import transcribe_audio, read_transcript