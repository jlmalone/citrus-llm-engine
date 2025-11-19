"""Item categorization endpoint."""

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.models.request import CategorizeRequest
from app.ml.categorization_engine import CategorizationEngine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/categorize")
async def categorize_items(request: CategorizeRequest) -> dict:
    """Categorize receipt items using ML."""
    try:
        categorizer = CategorizationEngine()

        results = []
        for item_desc in request.items:
            category_info = categorizer.categorize(
                item_description=item_desc,
                merchant=request.merchant_name,
            )

            # Filter by confidence threshold
            if category_info["confidence"] >= (
                request.confidence_threshold or 0.75
            ):
                results.append({
                    "item": item_desc,
                    "category": category_info["category"],
                    "confidence": category_info["confidence"],
                    "tax_deductible": category_info["tax_deductible"],
                    "tax_categories": {
                        "irs": category_info["irs_category"],
                        "cra": category_info["cra_category"],
                        "hmrc": category_info["hmrc_category"],
                    },
                })
            else:
                results.append({
                    "item": item_desc,
                    "category": "Uncategorized",
                    "confidence": category_info["confidence"],
                    "tax_deductible": False,
                    "tax_categories": None,
                })

        return {
            "success": True,
            "count": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Categorization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")
