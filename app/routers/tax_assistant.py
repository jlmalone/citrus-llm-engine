"""
Tax assistant endpoint for deduction identification
"""
import time
import logging
from fastapi import APIRouter, HTTPException
from app.models.request import TaxAssistantRequest
from app.models.response import TaxAssistantResponse, TaxDeduction
from app.ml.tax_assistant import tax_assistant
from app.services.llm_service import llm_service
from app.services.prompt_builder import prompt_builder
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["tax"])


@router.post("/tax-assistant", response_model=TaxAssistantResponse)
async def analyze_tax_deduction(request: TaxAssistantRequest):
    """
    Analyze expense for tax deductibility
    """
    start_time = time.time()

    try:
        tax_year = request.tax_year or settings.TAX_YEAR

        # First try rule-based tax assistant
        rule_result = await tax_assistant.analyze_deduction(
            description=request.description,
            amount=request.amount,
            category=request.category,
            region=request.region,
            tax_year=tax_year,
            business_use=request.business_use,
        )

        # If confidence is high, use rule-based result
        if rule_result["confidence"] >= 0.80:
            deduction = TaxDeduction(**rule_result)

            processing_time = int((time.time() - start_time) * 1000)

            return TaxAssistantResponse(
                success=True,
                region=request.region,
                tax_year=tax_year,
                deduction=deduction,
                confidence=rule_result["confidence"],
                reasoning=f"Matched to tax rule: {rule_result['category']}",
                warnings=[],
                references=[],
                processing_time_ms=processing_time,
                metadata={"method": "rule-based"},
            )

        # Fallback to LLM for complex cases
        logger.info("Using LLM for tax analysis")

        messages = prompt_builder.build_tax_assistant_prompt(
            description=request.description,
            amount=request.amount,
            category=request.category,
            region=request.region,
            tax_year=tax_year,
            business_use=request.business_use,
        )

        llm_result = await llm_service.extract_json(messages=messages)

        deduction_data = llm_result["parsed_json"]

        deduction = TaxDeduction(**deduction_data)

        processing_time = int((time.time() - start_time) * 1000)

        # Build warnings
        warnings = []
        if deduction.deduction_type == "conditional":
            warnings.append("This deduction has specific conditions that must be met")
        if deduction.deduction_type == "requires_documentation":
            warnings.append("Proper documentation is required to claim this deduction")

        # Build references (would normally come from a database)
        references = []
        if request.region == "US":
            references.append("IRS Publication 535 - Business Expenses")
        elif request.region == "CA":
            references.append("CRA Guide T4002 - Business and Professional Income")
        elif request.region == "GB":
            references.append("HMRC - Expenses if you're self-employed")

        return TaxAssistantResponse(
            success=True,
            region=request.region,
            tax_year=tax_year,
            deduction=deduction,
            confidence=0.85,
            reasoning=deduction.notes or "LLM-based analysis",
            warnings=warnings,
            references=references,
            processing_time_ms=processing_time,
            metadata={
                "method": "llm",
                "tokens_used": llm_result["usage"]["total_tokens"],
            },
        )

    except Exception as e:
        logger.error(f"Tax analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tax analysis failed: {str(e)}")
