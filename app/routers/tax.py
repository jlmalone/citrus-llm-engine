"""Tax analysis endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from app.models.request import TaxAnalysisRequest
from app.models.response import TaxAnalysisResponse, TaxDeduction
from app.ml.tax_assistant import TaxAssistant

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=TaxAnalysisResponse)
async def analyze_tax_deductions(request: TaxAnalysisRequest) -> TaxAnalysisResponse:
    """Analyze receipt for tax deductions."""
    try:
        tax_assistant = TaxAssistant()

        analysis = tax_assistant.analyze_receipt(
            merchant=request.merchant_name,
            items=request.items,
            total_amount=request.total_amount,
            jurisdiction=request.jurisdiction,
            date=request.date,
        )

        return TaxAnalysisResponse(
            jurisdiction=analysis["jurisdiction"],
            total_amount=analysis["total_amount"],
            deductible_amount=analysis["deductible_amount"],
            deductible_percentage=analysis["deductible_percentage"],
            deductions=[
                TaxDeduction(
                    item_description=d["item"],
                    amount=d["amount"],
                    deductible=d["deductible"],
                    deduction_type=d["category"],
                    percentage=d["percentage"],
                    notes=d.get("reasoning"),
                )
                for d in analysis["deductions"]
            ],
            recommendations=analysis["recommendations"],
            warnings=analysis["warnings"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Tax analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tax analysis failed: {str(e)}")
