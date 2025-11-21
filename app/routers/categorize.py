"""
Expense categorization endpoint
"""
import time
import logging
from fastapi import APIRouter, HTTPException
from app.models.request import CategorizationRequest
from app.models.response import CategorizationResponse, CategoryPrediction
from app.ml.categorization_engine import categorization_engine
from app.services.llm_service import llm_service
from app.services.prompt_builder import prompt_builder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["categorization"])


@router.post("/categorize", response_model=CategorizationResponse)
async def categorize_expense(request: CategorizationRequest):
    """
    Categorize expense with ML and LLM fallback
    """
    start_time = time.time()

    try:
        # Try ML categorization first
        if request.use_ml:
            result = await categorization_engine.categorize(
                description=request.description,
                merchant=request.merchant,
                amount=request.amount,
            )

            processing_time = int((time.time() - start_time) * 1000)

            # If confidence is high enough, return ML result
            if result["primary_category"]["confidence"] >= 0.75:
                return CategorizationResponse(
                    success=True,
                    primary_category=CategoryPrediction(**result["primary_category"]),
                    alternative_categories=[],
                    method="ml",
                    processing_time_ms=processing_time,
                    metadata={"tax_info": result.get("tax_info")},
                )

        # Fallback to LLM categorization
        if request.use_llm:
            logger.info("Using LLM for categorization")

            messages = prompt_builder.build_categorization_prompt(
                description=request.description,
                merchant=request.merchant,
                amount=request.amount,
            )

            llm_result = await llm_service.extract_json(messages=messages)

            category_data = llm_result["parsed_json"]

            primary = CategoryPrediction(
                category=category_data["category"],
                confidence=category_data["confidence"],
                tax_category=category_data.get("tax_category"),
                deductible=category_data.get("deductible"),
                subcategories=category_data.get("subcategories", []),
            )

            processing_time = int((time.time() - start_time) * 1000)

            return CategorizationResponse(
                success=True,
                primary_category=primary,
                alternative_categories=[],
                method="llm",
                processing_time_ms=processing_time,
                metadata={"tokens_used": llm_result["usage"]["total_tokens"]},
            )

        raise HTTPException(
            status_code=400, detail="Both ML and LLM categorization are disabled"
        )

    except Exception as e:
        logger.error(f"Categorization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")
