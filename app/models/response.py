"""
Response models for citrus-llm-engine API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class MerchantInfo(BaseModel):
    """Merchant information extracted from receipt."""

    name: str = Field(..., description="Normalized merchant name")
    store_number: Optional[str] = Field(None, description="Store number if available")
    address: Optional[str] = Field(None, description="Store address if available")


class ReceiptItem(BaseModel):
    """Individual item from receipt."""

    description: str = Field(..., description="Item description")
    price: float = Field(..., description="Item price", ge=0)
    quantity: int = Field(1, description="Item quantity", ge=1)
    category: str = Field(..., description="Inferred item category")


class ReceiptTotals(BaseModel):
    """Receipt totals."""

    subtotal: float = Field(..., description="Subtotal before tax", ge=0)
    tax: float = Field(..., description="Tax amount", ge=0)
    total: float = Field(..., description="Total amount", ge=0)


class ReceiptExtraction(BaseModel):
    """Complete receipt extraction data."""

    merchant: MerchantInfo = Field(..., description="Merchant information")
    timestamp: Optional[str] = Field(None, description="Receipt timestamp in ISO8601 format")
    items: list[ReceiptItem] = Field(..., description="List of receipt items")
    totals: ReceiptTotals = Field(..., description="Receipt totals")
    payment_method: Optional[str] = Field(None, description="Payment method if available")
    currency: str = Field("USD", description="Currency code")


class ExtractionResponse(BaseModel):
    """Response model for /api/extract endpoint."""

    success: bool = Field(..., description="Whether extraction was successful")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    extraction: Optional[ReceiptExtraction] = Field(None, description="Extracted receipt data")
    error: Optional[str] = Field(None, description="Error message if extraction failed")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "confidence": 0.95,
                    "extraction": {
                        "merchant": {
                            "name": "Walmart",
                            "store_number": "1234",
                            "address": "123 Main St"
                        },
                        "timestamp": "2024-12-25T15:45:00Z",
                        "items": [
                            {
                                "description": "Milk 2% Gallon",
                                "price": 3.99,
                                "quantity": 1,
                                "category": "Groceries > Dairy"
                            },
                            {
                                "description": "Bread Wheat",
                                "price": 2.49,
                                "quantity": 1,
                                "category": "Groceries > Bakery"
                            },
                            {
                                "description": "Eggs Dozen",
                                "price": 4.29,
                                "quantity": 1,
                                "category": "Groceries > Dairy"
                            }
                        ],
                        "totals": {
                            "subtotal": 10.77,
                            "tax": 0.75,
                            "total": 11.52
                        },
                        "payment_method": None,
                        "currency": "USD"
                    }
                }
            ]
        }
    }


class ChatChoice(BaseModel):
    """OpenAI-compatible chat choice."""

    index: int
    message: dict
    finish_reason: str


class ChatUsage(BaseModel):
    """OpenAI-compatible usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
