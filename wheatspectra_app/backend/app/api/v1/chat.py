"""
PhytoSpectra-VLM / WheatSpectraChat
Chat conversation API endpoint.
"""

from fastapi import APIRouter, HTTPException, status
from app.core.gpu_lock import get_gpu_lock
from app.core.registry import get_model_registry
from app.schemas.api import ChatRequest, ChatResponse
from app.services.chat_pipeline import ChatPipeline

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def post_chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    registry = get_model_registry()
    pipeline = ChatPipeline(registry)

    async with get_gpu_lock():
        try:
            response = pipeline.process_chat(req)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The scientific reasoning pipeline could not complete this request.",
            )

    return response
