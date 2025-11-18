"""
OpenAI-compatible chat completions endpoint.
Provides maximum flexibility for custom prompts and integrations.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models.request import ChatCompletionRequest
from app.models.response import ChatCompletionResponse
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint.

    This endpoint provides full compatibility with the OpenAI API,
    allowing drop-in replacement for OpenAI API calls with local LLMs.

    Supports:
    - LM Studio (local)
    - Ollama (local)
    - OpenAI (cloud fallback)

    Args:
        request: Chat completion request in OpenAI format

    Returns:
        Chat completion response in OpenAI format

    Raises:
        HTTPException: If completion fails
    """
    logger.info(f"Processing chat completion: model={request.model}, messages={len(request.messages)}")

    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not yet supported"
        )

    try:
        # Initialize LLM service
        llm_service = LLMService(model=request.model)

        # Convert Pydantic models to dicts for LLM service
        messages = [msg.model_dump() for msg in request.messages]

        # Call LLM
        completion = await llm_service.chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        logger.info(f"Chat completion successful: {completion['usage']['total_tokens']} tokens")

        return ChatCompletionResponse(**completion)

    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat completion failed: {str(e)}"
        )
