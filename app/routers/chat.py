"""OpenAI-compatible chat completion endpoint."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from app.models.request import ChatCompletionRequest
from app.models.response import ChatCompletionResponse
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Create a chat completion (OpenAI-compatible endpoint).

    This endpoint is fully compatible with the OpenAI chat completions API.

    Args:
        request: Chat completion request

    Returns:
        Chat completion response

    Raises:
        HTTPException: If the completion fails
    """
    logger.info(f"Chat completion request for model: {request.model}")

    # Validate streaming not supported yet
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming is not yet supported",
        )

    # Validate n parameter
    if request.n and request.n > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Multiple completions (n > 1) not yet supported",
        )

    try:
        # Initialize LLM service
        llm_service = LLMService()

        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Create completion
        response = await llm_service.chat_completion(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        logger.info(f"Chat completion successful for model: {request.model}")
        return response

    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}",
        )
