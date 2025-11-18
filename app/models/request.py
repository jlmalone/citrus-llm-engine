"""Request models for the Citrus LLM Engine."""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request model for receipt extraction."""

    ocr_text: str = Field(..., description="OCR text from the receipt")
    image_url: Optional[str] = Field(None, description="Optional image URL for vision models")
    model: Optional[str] = Field(None, description="Override the default LLM model")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature for LLM")
    max_tokens: Optional[int] = Field(None, gt=0, description="Max tokens for LLM response")


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2000, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    n: Optional[int] = Field(1, description="Number of completions to generate")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    logit_bias: Optional[Dict[str, float]] = Field(None, description="Logit bias")
    user: Optional[str] = Field(None, description="User identifier")
