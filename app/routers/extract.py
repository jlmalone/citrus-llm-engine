"""
Receipt extraction endpoint
"""
import json
import time
import logging
from fastapi import APIRouter, HTTPException
from app.models.request import ExtractionRequest
from app.models.response import (
    ExtractionResponse,
    ReceiptExtraction,
    MerchantInfo,
    LineItem,
    Totals,
)
from app.services.llm_service import llm_service
from app.services.prompt_builder import prompt_builder
from app.services.ocr_service import ocr_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["extraction"])


@router.post("/extract", response_model=ExtractionResponse)
async def extract_receipt(request: ExtractionRequest):
    """
    Extract structured data from receipt OCR text
    """
    start_time = time.time()

    try:
        # Get OCR text (either from request or by processing image)
        ocr_text = request.ocr_text

        if request.use_ocr and (request.image_url or request.image_data):
            # Run OCR on image
            logger.info("Running OCR on provided image")
            image = await ocr_service.load_image(
                image_url=request.image_url, image_data=request.image_data
            )

            engines = request.ocr_engines or settings.OCR_ENGINES.split(",")
            languages = [request.language] if request.language else ["eng"]

            ocr_result = await ocr_service.extract_text_ensemble(
                image=image,
                engine_names=engines,
                languages=languages,
                preprocess=True,
                ensemble_mode="voting",
            )

            ocr_text = ocr_result["text"]
            logger.info(f"OCR extracted text: {len(ocr_text)} characters")

        # Build extraction prompt
        messages = prompt_builder.build_extraction_prompt(
            ocr_text=ocr_text, language=request.language or "en"
        )

        # Call LLM
        result = await llm_service.extract_json(
            messages=messages,
            provider=request.provider,
            model=request.model,
        )

        # Parse extraction result
        extraction_data = result["parsed_json"]

        # Build response
        merchant = MerchantInfo(**extraction_data.get("merchant", {}))

        items = [LineItem(**item) for item in extraction_data.get("items", [])]

        totals = Totals(**extraction_data.get("totals", {}))

        extraction = ReceiptExtraction(
            merchant=merchant,
            timestamp=extraction_data.get("timestamp"),
            items=items,
            totals=totals,
            payment_method=extraction_data.get("payment_method"),
            currency=extraction_data.get("currency", "USD"),
            receipt_number=extraction_data.get("receipt_number"),
            tax_id=extraction_data.get("tax_id"),
        )

        # Calculate confidence (heuristic)
        confidence = 0.9 if len(items) > 0 and totals.total > 0 else 0.6

        processing_time = int((time.time() - start_time) * 1000)

        return ExtractionResponse(
            success=True,
            confidence=confidence,
            extraction=extraction,
            provider=result["provider"],
            model=result["model"],
            processing_time_ms=processing_time,
            metadata={
                "tokens_used": result["usage"]["total_tokens"],
            },
        )

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
