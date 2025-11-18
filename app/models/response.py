"""Response models for the Citrus LLM Engine."""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


class MerchantInfo(BaseModel):
    """Merchant information extracted from receipt."""

    name: str = Field(..., description="Merchant name")
    store_number: Optional[str] = Field(None, description="Store number if available")
    address: Optional[str] = Field(None, description="Store address if available")


class ReceiptItem(BaseModel):
    """Individual item from receipt."""

    description: str = Field(..., description="Item description")
    price: float = Field(..., description="Item price")
    quantity: int = Field(1, description="Item quantity")
    category: Optional[str] = Field(None, description="Item category")


class ReceiptTotals(BaseModel):
    """Receipt totals information."""

    subtotal: float = Field(..., description="Subtotal amount")
    tax: float = Field(0.0, description="Tax amount")
    total: float = Field(..., description="Total amount")
    tip: Optional[float] = Field(None, description="Tip amount if applicable")
    discount: Optional[float] = Field(None, description="Discount amount if applicable")


class ReceiptExtraction(BaseModel):
    """Complete receipt extraction data."""

    merchant: MerchantInfo = Field(..., description="Merchant information")
    timestamp: Optional[str] = Field(None, description="Receipt timestamp in ISO8601 format")
    items: List[ReceiptItem] = Field(..., description="List of items")
    totals: ReceiptTotals = Field(..., description="Total amounts")
    payment_method: Optional[str] = Field(None, description="Payment method if available")
    currency: str = Field("USD", description="Currency code")


class ExtractionResponse(BaseModel):
    """Response model for receipt extraction."""

    success: bool = Field(..., description="Whether extraction was successful")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    extraction: Optional[ReceiptExtraction] = Field(None, description="Extracted data")
    error: Optional[str] = Field(None, description="Error message if failed")
    raw_response: Optional[str] = Field(None, description="Raw LLM response for debugging")


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = Field(..., description="Choice index")
    message: dict = Field(..., description="Message content")
    finish_reason: str = Field(..., description="Reason for completion finish")


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Tokens in prompt")
    completion_tokens: int = Field(..., description="Tokens in completion")
    total_tokens: int = Field(..., description="Total tokens")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(..., description="Completion ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatCompletionChoice] = Field(..., description="Completion choices")
    usage: ChatCompletionUsage = Field(..., description="Token usage")
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")
