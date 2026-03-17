# core/__init__.py

"""
Core module — configuration, Gemini client, token tracking, utilities.
"""

from core.config import config
from core.gemini_client import gemini_client
from core.token_tracker import TokenTracker, reset_tracker
from core.utils import redact_pii_text, redact_pii_from_image, enforce_tone