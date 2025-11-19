"""OpenAI-compatible chat completions endpoint."""

import logging
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.request import ChatCompletionRequest
from app.models.response import ChatCompletionResponse
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    try:
        llm_service = LLMService()

        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Get completion
        response_text = await llm_service.provider.complete(
            messages=messages,
            temperature=request.temperature or 0.1,
            max_tokens=request.max_tokens or 2048,
        )

        # Build OpenAI-compatible response
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(datetime.utcnow().timestamp()),
            model=llm_service.get_model_name(),
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": sum(len(m["content"].split()) for m in messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": sum(len(m["content"].split()) for m in messages)
                + len(response_text.split()),
            },
        )

    except Exception as e:
        logger.error(f"Chat completion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")
