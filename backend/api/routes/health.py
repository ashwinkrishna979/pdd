"""Health check route."""

from fastapi import APIRouter

from core.gemini_client import gemini_client
from core.config import config
from api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    configured = gemini_client.is_configured()
    available = gemini_client.is_available() if configured else False
    error = gemini_client.last_health_error() if not available else None
    return HealthResponse(
        status="ok",
        gemini_configured=configured,
        gemini_available=available,
        gemini_model=config.gemini.text_model if configured else None,
        gemini_error=error or None,
    )
