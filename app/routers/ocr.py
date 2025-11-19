"""OCR processing endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from app.models.request import OCRRequest
from app.models.response import OCRResponse
from app.ml.ocr_engine import OCRService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ocr", response_model=OCRResponse)
async def process_ocr(request: OCRRequest) -> OCRResponse:
    """Process OCR on image or PDF."""
    try:
        # Determine image source
        image_source = request.image_url or request.image_path or request.pdf_url

        if not image_source:
            raise HTTPException(
                status_code=400,
                detail="One of image_url, image_path, or pdf_url must be provided",
            )

        # Initialize OCR service
        ocr_service = OCRService(engine=request.engine)

        # Process OCR
        result = await ocr_service.extract_text(
            image_data=image_source,
            preprocessing=request.preprocessing if request.preprocessing is not None else True,
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"OCR processing failed: {result.get('error', 'Unknown error')}",
            )

        return OCRResponse(
            success=True,
            text=result["text"],
            confidence=result["confidence"],
            engine=result["engine"],
            processing_time_ms=result["processing_time_ms"],
            metadata=result.get("metadata"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
