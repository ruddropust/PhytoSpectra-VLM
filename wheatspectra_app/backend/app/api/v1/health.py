"""
PhytoSpectra-VLM / WheatSpectraChat
Health API endpoint.
"""

from fastapi import APIRouter
from app.config import settings
from app.schemas.api import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        project=settings.PROJECT_NAME,
    )
