"""
OpenAI-compatible chat completions endpoint
"""
import time
import uuid
import logging
from fastapi import APIRouter, HTTPException
from app.models.request import ChatCompletionRequest
from app.models.response import (
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
)
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint
    """
    try:
        # Convert messages to dict format
        messages = [msg.model_dump() for msg in request.messages]

        # Call LLM service
        result = await llm_service.complete(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=result["model"],
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": result["content"]},
                    finish_reason=result["finish_reason"],
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=result["usage"]["prompt_tokens"],
                completion_tokens=result["usage"]["completion_tokens"],
                total_tokens=result["usage"]["total_tokens"],
            ),
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Chat completion failed: {str(e)}"
        )
