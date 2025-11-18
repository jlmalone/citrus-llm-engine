"""Receipt extraction endpoint."""

import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from app.models.request import ExtractionRequest
from app.models.response import ExtractionResponse, ReceiptExtraction
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["extraction"])


@router.post("/extract", response_model=ExtractionResponse)
async def extract_receipt(request: ExtractionRequest) -> Dict[str, Any]:
    """Extract structured data from receipt OCR text.

    This is a convenience endpoint that wraps the LLM service with
    specialized prompts for receipt extraction.

    Args:
        request: Extraction request with OCR text

    Returns:
        Extraction response with structured data

    Raises:
        HTTPException: If extraction fails
    """
    logger.info("Receipt extraction request received")

    try:
        # Build messages for LLM
        messages = PromptBuilder.build_messages(request.ocr_text)

        # Initialize LLM service
        llm_service = LLMService()

        # Extract receipt data
        raw_response = await llm_service.extract_receipt(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        logger.debug(f"Raw LLM response: {raw_response}")

        # Parse JSON response
        try:
            extraction_data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw response: {raw_response}")
            return {
                "success": False,
                "confidence": 0.0,
                "extraction": None,
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
                "raw_response": raw_response,
            }

        # Validate extraction data structure
        try:
            extraction = ReceiptExtraction(**extraction_data)
            logger.info("Receipt extraction successful")

            return {
                "success": True,
                "confidence": 0.95,  # Could be made dynamic based on LLM response
                "extraction": extraction.model_dump(),
                "error": None,
                "raw_response": raw_response,
            }

        except Exception as e:
            logger.error(f"Failed to validate extraction data: {str(e)}")
            return {
                "success": False,
                "confidence": 0.0,
                "extraction": None,
                "error": f"Failed to validate extraction data: {str(e)}",
                "raw_response": raw_response,
            }

    except Exception as e:
        logger.error(f"Receipt extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Receipt extraction failed: {str(e)}",
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status information
    """
    logger.info("Health check requested")

    try:
        # Check LLM service health
        llm_service = LLMService()
        llm_healthy = await llm_service.health_check()

        return {
            "status": "healthy" if llm_healthy else "degraded",
            "llm_provider": llm_service.provider,
            "llm_healthy": llm_healthy,
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }
