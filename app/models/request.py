"""
Request models for citrus-llm-engine API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request model for /api/extract endpoint."""

    ocr_text: str = Field(
        ...,
        description="OCR text extracted from receipt",
        min_length=1
    )
    image_url: Optional[str] = Field(
        None,
        description="Optional image URL for vision models"
    )
    model: Optional[str] = Field(
        None,
        description="Optional model override (uses default from config if not specified)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ocr_text": "WAL*MART #1234\n123 Main St\n12/25/2024 3:45 PM\n\nMILK 2% GAL    $3.99\nBREAD WHEAT    $2.49\nEGGS DOZEN     $4.29\n\nSUBTOTAL      $10.77\nTAX            $0.75\nTOTAL         $11.52",
                    "model": "lmstudio-local"
                }
            ]
        }
    }


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model to use for completion")
    messages: list[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2000, ge=1)
    stream: Optional[bool] = Field(False, description="Whether to stream responses")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            ]
        }
    }
