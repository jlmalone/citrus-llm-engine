"""Response models for API endpoints."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Merchant(BaseModel):
    """Merchant information extracted from receipt."""

    name: str
    store_number: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None


class Item(BaseModel):
    """Receipt line item."""

    description: str
    price: float
    quantity: int = 1
    category: Optional[str] = None
    tax_deductible: Optional[bool] = None
    tax_category: Optional[str] = None
    confidence: Optional[float] = None


class Totals(BaseModel):
    """Receipt totals."""

    subtotal: float
    tax: float
    total: float
    tip: Optional[float] = None
    discount: Optional[float] = None


class TaxCategory(BaseModel):
    """Tax category classification."""

    category: str
    confidence: float
    subcategory: Optional[str] = None
    description: Optional[str] = None


class TaxDeduction(BaseModel):
    """Tax deduction information."""

    item_description: str
    amount: float
    deductible: bool
    deduction_type: Optional[str] = None
    percentage: float = 100.0
    notes: Optional[str] = None
    irs_category: Optional[str] = None  # For US
    cra_category: Optional[str] = None  # For Canada
    hmrc_category: Optional[str] = None  # For UK


class TaxAnalysisResponse(BaseModel):
    """Tax analysis response."""

    jurisdiction: str
    total_amount: float
    deductible_amount: float
    deductible_percentage: float
    deductions: List[TaxDeduction]
    recommendations: List[str]
    warnings: List[str] = []


class OCRResponse(BaseModel):
    """OCR processing response."""

    success: bool
    text: str
    confidence: float
    engine: str
    language: Optional[str] = None
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class ExtractResponse(BaseModel):
    """Receipt extraction response."""

    success: bool
    confidence: float
    extraction: Dict[str, Any] = Field(
        ...,
        description="Extracted data including merchant, items, totals",
    )
    ocr_result: Optional[OCRResponse] = None
    tax_analysis: Optional[TaxAnalysisResponse] = None
    processing_time_ms: float
    model_used: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime
    services: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of dependent services",
    )
