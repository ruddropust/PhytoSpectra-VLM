"""
PhytoSpectra-VLM / WheatSpectraChat
API v1 router aggregator.
"""

from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.model import router as model_router
from app.api.v1.session import router as session_router
from app.api.v1.upload import router as upload_router
from app.api.v1.chat import router as chat_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router, tags=["Health"])
api_v1_router.include_router(model_router, tags=["Model"])
api_v1_router.include_router(session_router, tags=["Session"])
api_v1_router.include_router(upload_router, tags=["Upload"])
api_v1_router.include_router(chat_router, tags=["Chat"])
