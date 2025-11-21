"""
OCR endpoint for text extraction from images
"""
import time
import logging
from fastapi import APIRouter, HTTPException
from app.models.request import OCRRequest
from app.models.response import OCRResponse, OCREngineResult
from app.services.ocr_service import ocr_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ocr"])


@router.post("/ocr", response_model=OCRResponse)
async def extract_text_ocr(request: OCRRequest):
    """
    Extract text from image using advanced OCR engines
    """
    start_time = time.time()

    try:
        # Load image
        image = await ocr_service.load_image(
            image_url=request.image_url, image_data=request.image_data
        )

        # Get engines to use
        engines = request.engines or settings.OCR_ENGINES.split(",")
        languages = request.languages or settings.OCR_LANGUAGES.split(",")

        # Run OCR ensemble
        result = await ocr_service.extract_text_ensemble(
            image=image,
            engine_names=engines,
            languages=languages,
            preprocess=request.preprocessing,
            ensemble_mode=request.ensemble_mode,
        )

        # Build response
        engine_results = [
            OCREngineResult(
                engine=r["engine"],
                text=r["text"],
                confidence=r["confidence"],
                processing_time_ms=r["processing_time_ms"],
                metadata=r.get("metadata"),
            )
            for r in result["engine_results"]
        ]

        processing_time = int((time.time() - start_time) * 1000)

        return OCRResponse(
            success=True,
            text=result["text"],
            confidence=result["confidence"],
            engines_used=result["engines_used"],
            engine_results=engine_results,
            ensemble_mode=result["ensemble_mode"],
            preprocessing_applied=request.preprocessing,
            processing_time_ms=processing_time,
            metadata={"character_count": len(result["text"])},
        )

    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
