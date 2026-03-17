# pipeline/__init__.py

"""
Pipeline orchestrators for PDD generation.
- audio_pipeline: Meeting recordings with audio → PDD
- video_pipeline: Silent screen recordings → PDD
- common: Shared pipeline logic
"""

from pipeline.audio_pipeline import AudioPipeline
from pipeline.video_pipeline import VideoPipeline