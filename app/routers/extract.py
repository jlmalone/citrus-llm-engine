"""
Receipt extraction endpoint.
Convenience wrapper for extracting structured data from receipt OCR text.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models.request import ExtractionRequest
from app.models.response import ExtractionResponse, ReceiptExtraction
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["extraction"])


@router.post("/extract", response_model=ExtractionResponse)
async def extract_receipt(request: ExtractionRequest) -> ExtractionResponse:
    """
    Extract structured data from receipt OCR text.

    This is a convenience endpoint that:
    1. Takes OCR text (and optional image URL)
    2. Builds an optimized prompt for receipt extraction
    3. Calls the configured LLM provider
    4. Parses and validates the response
    5. Returns structured receipt data

    Args:
        request: Extraction request with OCR text

    Returns:
        Structured receipt data with confidence score

    Raises:
        HTTPException: If extraction fails
    """
    logger.info(f"Processing extraction request (text length: {len(request.ocr_text)} chars)")

    try:
        # Initialize LLM service with optional model override
        llm_service = LLMService(model=request.model)

        # Build extraction messages
        messages = PromptBuilder.build_chat_messages(
            ocr_text=request.ocr_text,
            image_url=request.image_url
        )

        # Call LLM
        completion = await llm_service.chat_completion(messages)

        # Extract and parse response
        response_text = llm_service.extract_text_from_completion(completion)
        extraction_data = llm_service.parse_json_response(response_text)

        # Validate and create response
        extraction = ReceiptExtraction(**extraction_data)

        # Calculate confidence (simplified - could be more sophisticated)
        confidence = _calculate_confidence(extraction)

        logger.info(f"Extraction successful (confidence: {confidence:.2f})")

        return ExtractionResponse(
            success=True,
            confidence=confidence,
            extraction=extraction
        )

    except ValueError as e:
        # JSON parsing or validation error
        logger.error(f"Extraction validation failed: {str(e)}")
        return ExtractionResponse(
            success=False,
            confidence=0.0,
            error=f"Failed to parse LLM response: {str(e)}"
        )

    except Exception as e:
        # LLM or other error
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


def _calculate_confidence(extraction: ReceiptExtraction) -> float:
    """
    Calculate confidence score for extraction.

    Simple heuristic based on:
    - Presence of merchant name
    - Presence of items
    - Totals matching (if we can calculate)
    - Timestamp presence

    Args:
        extraction: The extracted receipt data

    Returns:
        Confidence score between 0 and 1
    """
    score = 0.0
    checks = 0

    # Check merchant name
    checks += 1
    if extraction.merchant.name and len(extraction.merchant.name) > 2:
        score += 1.0

    # Check items
    checks += 1
    if extraction.items and len(extraction.items) > 0:
        score += 1.0

    # Check totals
    checks += 1
    if extraction.totals.total > 0:
        score += 1.0

        # Verify totals match (subtotal + tax should equal total, within small margin)
        expected_total = extraction.totals.subtotal + extraction.totals.tax
        if abs(expected_total - extraction.totals.total) < 0.02:
            score += 0.5
            checks += 0.5

    # Check timestamp
    checks += 1
    if extraction.timestamp:
        score += 1.0

    # Return normalized confidence
    return min(score / checks, 1.0) if checks > 0 else 0.0
