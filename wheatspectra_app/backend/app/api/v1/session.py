"""
PhytoSpectra-VLM / WheatSpectraChat
Session creation and management API endpoints.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.session_cache import session_cache
from app.schemas.api import SessionResponse

router = APIRouter()


@router.post("/session", response_model=SessionResponse)
async def create_session():
    session_id = session_cache.create_session()
    now_iso = datetime.now(timezone.utc).isoformat()
    return SessionResponse(session_id=session_id, created_at=now_iso)
