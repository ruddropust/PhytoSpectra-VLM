"""
PhytoSpectra-VLM / WheatSpectraChat
Model status API endpoint. Exposes loaded models, device, and parameter counts without sensitive local filesystem paths.
"""

from fastapi import APIRouter
from app.core.registry import get_model_registry
from app.schemas.api import ModelStatusResponse
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers"
)

router = APIRouter()


@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    registry = get_model_registry()
    status = registry.get_status()
    return ModelStatusResponse(**status)
