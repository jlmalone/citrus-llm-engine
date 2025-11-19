"""Request models for API endpoints."""

from typing import Optional, List, Literal, Any, Dict
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message for OpenAI-compatible interface."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False


class OCRRequest(BaseModel):
    """Request for OCR processing."""

    image_url: Optional[str] = Field(
        None, description="URL or base64-encoded image data"
    )
    image_path: Optional[str] = Field(None, description="Local file path to image")
    pdf_url: Optional[str] = Field(None, description="URL or path to PDF file")
    engine: Optional[Literal["tesseract", "easyocr", "paddleocr", "ensemble"]] = Field(
        None, description="OCR engine to use (defaults to config)"
    )
    languages: Optional[List[str]] = Field(
        None, description="Languages to detect (e.g., ['en', 'fr'])"
    )
    preprocessing: Optional[bool] = Field(
        True, description="Apply image preprocessing"
    )
    confidence_threshold: Optional[float] = Field(
        0.8, description="Minimum confidence threshold"
    )


class ExtractRequest(BaseModel):
    """Request for receipt extraction."""

    ocr_text: Optional[str] = Field(
        None, description="Pre-extracted OCR text from receipt"
    )
    image_url: Optional[str] = Field(
        None, description="URL or base64 image (will run OCR first)"
    )
    image_path: Optional[str] = Field(None, description="Local image path")
    pdf_url: Optional[str] = Field(None, description="PDF document")
    model: Optional[str] = Field(None, description="LLM model to use")
    enable_categorization: Optional[bool] = Field(
        True, description="Enable ML categorization"
    )
    enable_tax_analysis: Optional[bool] = Field(
        True, description="Enable tax deduction analysis"
    )
    tax_jurisdiction: Optional[Literal["US", "CA", "GB"]] = Field(
        None, description="Tax jurisdiction for analysis"
    )


class CategorizeRequest(BaseModel):
    """Request for ML-based categorization."""

    items: List[str] = Field(..., description="List of item descriptions")
    merchant_name: Optional[str] = Field(None, description="Merchant name for context")
    confidence_threshold: Optional[float] = Field(
        0.75, description="Minimum confidence for categorization"
    )


class TaxAnalysisRequest(BaseModel):
    """Request for tax deduction analysis."""

    merchant_name: str
    items: List[Dict[str, Any]] = Field(
        ..., description="List of items with description, price, category"
    )
    total_amount: float
    date: Optional[str] = Field(None, description="Receipt date (ISO format)")
    jurisdiction: Literal["US", "CA", "GB"] = Field(
        default="US", description="Tax jurisdiction"
    )
    taxpayer_type: Optional[Literal["individual", "business"]] = Field(
        default="business", description="Taxpayer type"
    )
