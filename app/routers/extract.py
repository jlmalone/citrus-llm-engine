"""Receipt extraction endpoint."""

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from app.models.request import ExtractRequest
from app.models.response import ExtractResponse, OCRResponse, TaxAnalysisResponse
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.ml.ocr_engine import OCRService
from app.ml.categorization_engine import CategorizationEngine
from app.ml.tax_assistant import TaxAssistant
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/extract", response_model=ExtractResponse)
async def extract_receipt(request: ExtractRequest) -> ExtractResponse:
    """Extract structured data from receipt."""
    start_time = time.time()

    try:
        ocr_result = None
        ocr_text = request.ocr_text

        # Step 1: OCR if image provided
        if not ocr_text and (request.image_url or request.image_path or request.pdf_url):
            logger.info("Running OCR on provided image")
            ocr_service = OCRService()

            image_source = request.image_url or request.image_path or request.pdf_url
            ocr_data = await ocr_service.extract_text(image_source)

            if not ocr_data.get("success"):
                raise HTTPException(
                    status_code=400,
                    detail=f"OCR failed: {ocr_data.get('error', 'Unknown error')}",
                )

            ocr_text = ocr_data["text"]
            ocr_result = OCRResponse(
                success=True,
                text=ocr_text,
                confidence=ocr_data["confidence"],
                engine=ocr_data["engine"],
                processing_time_ms=ocr_data["processing_time_ms"],
                metadata=ocr_data.get("metadata"),
            )

        if not ocr_text:
            raise HTTPException(
                status_code=400,
                detail="Either ocr_text or image must be provided",
            )

        # Step 2: LLM extraction
        logger.info("Extracting structured data with LLM")
        llm_service = LLMService(provider=request.model)
        system_prompt = PromptBuilder.get_extraction_system_prompt()

        extraction_data = await llm_service.extract_receipt_data(
            ocr_text=ocr_text,
            system_prompt=system_prompt,
        )

        # Step 3: ML categorization (if enabled)
        if request.enable_categorization and settings.enable_auto_categorization:
            logger.info("Applying ML categorization")
            categorizer = CategorizationEngine()

            items = extraction_data.get("items", [])
            merchant_name = extraction_data.get("merchant", {}).get("name")

            for item in items:
                description = item.get("description", "")
                if description:
                    category_info = categorizer.categorize(description, merchant_name)
                    item["category"] = category_info["category"]
                    item["tax_deductible"] = category_info["tax_deductible"]
                    item["confidence"] = category_info["confidence"]

        # Step 4: Tax analysis (if enabled)
        tax_analysis = None
        if request.enable_tax_analysis and settings.enable_tax_assistant:
            logger.info("Running tax deduction analysis")
            tax_assistant = TaxAssistant()

            jurisdiction = request.tax_jurisdiction or settings.default_tax_jurisdiction
            merchant_name = extraction_data.get("merchant", {}).get("name", "Unknown")
            items = extraction_data.get("items", [])
            total_amount = extraction_data.get("totals", {}).get("total", 0.0)
            date = extraction_data.get("timestamp")

            tax_data = tax_assistant.analyze_receipt(
                merchant=merchant_name,
                items=items,
                total_amount=total_amount,
                jurisdiction=jurisdiction,
                date=date,
            )

            tax_analysis = TaxAnalysisResponse(
                jurisdiction=tax_data["jurisdiction"],
                total_amount=tax_data["total_amount"],
                deductible_amount=tax_data["deductible_amount"],
                deductible_percentage=tax_data["deductible_percentage"],
                deductions=[
                    {
                        "item_description": d["item"],
                        "amount": d["amount"],
                        "deductible": d["deductible"],
                        "deduction_type": d["category"],
                        "percentage": d["percentage"],
                        "notes": d.get("notes"),
                    }
                    for d in tax_data["deductions"]
                ],
                recommendations=tax_data["recommendations"],
                warnings=tax_data["warnings"],
            )

        processing_time = (time.time() - start_time) * 1000

        # Calculate overall confidence
        confidence = 0.85  # Base confidence
        if ocr_result:
            confidence = (confidence + ocr_result.confidence) / 2

        return ExtractResponse(
            success=True,
            confidence=confidence,
            extraction=extraction_data,
            ocr_result=ocr_result,
            tax_analysis=tax_analysis,
            processing_time_ms=processing_time,
            model_used=llm_service.get_model_name(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
