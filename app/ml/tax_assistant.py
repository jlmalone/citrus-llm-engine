"""
Tax Assistant for IRS, CRA, and HMRC deduction identification
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.models.response import TaxDeductionType

logger = logging.getLogger(__name__)


class TaxRule:
    """Represents a tax deduction rule"""

    def __init__(
        self,
        category: str,
        deduction_type: TaxDeductionType,
        percentage: float,
        conditions: List[str],
        documentation: List[str],
        notes: str,
    ):
        self.category = category
        self.deduction_type = deduction_type
        self.percentage = percentage
        self.conditions = conditions
        self.documentation = documentation
        self.notes = notes


class IRSTaxRules:
    """IRS (US) tax rules"""

    @staticmethod
    def get_rules() -> Dict[str, TaxRule]:
        """Get IRS tax rules"""
        return {
            "Office Supplies": TaxRule(
                category="Business Expenses",
                deduction_type=TaxDeductionType.FULLY_DEDUCTIBLE,
                percentage=100.0,
                conditions=["Must be ordinary and necessary for business"],
                documentation=["Receipts", "Business purpose documentation"],
                notes="Fully deductible business expenses under Schedule C",
            ),
            "Software": TaxRule(
                category="Business Expenses",
                deduction_type=TaxDeductionType.FULLY_DEDUCTIBLE,
                percentage=100.0,
                conditions=["Must be used for business purposes"],
                documentation=["Receipts", "Proof of business use"],
                notes="Business software subscriptions are fully deductible",
            ),
            "Meals": TaxRule(
                category="Meals and Entertainment",
                deduction_type=TaxDeductionType.PARTIALLY_DEDUCTIBLE,
                percentage=50.0,
                conditions=[
                    "Must be business-related",
                    "Cannot be lavish or extravagant",
                    "You or employee must be present",
                ],
                documentation=["Receipt", "Business purpose", "People present"],
                notes="50% deductible for business meals",
            ),
            "Vehicle": TaxRule(
                category="Vehicle Expenses",
                deduction_type=TaxDeductionType.PARTIALLY_DEDUCTIBLE,
                percentage=None,  # Based on business use %
                conditions=["Must keep mileage log", "Business use only"],
                documentation=[
                    "Mileage log",
                    "Business purpose for each trip",
                    "Receipts if actual expenses",
                ],
                notes="Deductible based on business use percentage. Standard mileage rate or actual expenses.",
            ),
            "Home Office": TaxRule(
                category="Home Office",
                deduction_type=TaxDeductionType.CONDITIONAL,
                percentage=None,
                conditions=[
                    "Exclusive and regular business use",
                    "Principal place of business",
                ],
                documentation=["Square footage calculation", "Home expenses"],
                notes="Deductible based on percentage of home used exclusively for business (Form 8829)",
            ),
            "Travel": TaxRule(
                category="Travel Expenses",
                deduction_type=TaxDeductionType.FULLY_DEDUCTIBLE,
                percentage=100.0,
                conditions=[
                    "Must be away from tax home",
                    "Must be primarily for business",
                ],
                documentation=[
                    "Receipts",
                    "Business purpose",
                    "Dates and locations",
                ],
                notes="Transportation, lodging, and 50% of meals while traveling for business",
            ),
            "Medical": TaxRule(
                category="Medical Expenses",
                deduction_type=TaxDeductionType.CONDITIONAL,
                percentage=None,
                conditions=["Must exceed 7.5% of AGI"],
                documentation=["Medical receipts", "Insurance statements"],
                notes="Only amounts exceeding 7.5% of Adjusted Gross Income (Schedule A)",
            ),
            "Education": TaxRule(
                category="Education Expenses",
                deduction_type=TaxDeductionType.CONDITIONAL,
                percentage=100.0,
                conditions=[
                    "Maintains or improves job skills",
                    "Required by employer or law",
                    "Not for new trade/business",
                ],
                documentation=["Tuition receipts", "Course description"],
                notes="Work-related education expenses (Form 8863 or Schedule A)",
            ),
            "Advertising": TaxRule(
                category="Advertising",
                deduction_type=TaxDeductionType.FULLY_DEDUCTIBLE,
                percentage=100.0,
                conditions=["Must be reasonable and related to business"],
                documentation=["Receipts", "Advertising materials/proof"],
                notes="Business advertising and marketing expenses",
            ),
            "Professional Services": TaxRule(
                category="Professional Services",
                deduction_type=TaxDeductionType.FULLY_DEDUCTIBLE,
                percentage=100.0,
                conditions=["Must be for business purposes"],
                documentation=["Invoices", "Service contracts"],
                notes="Legal, accounting, consulting fees for business",
            ),
        }


class CRATaxRules:
    """CRA (Canada) tax rules"""

    @staticmethod
    def get_rules() -> Dict[str, TaxRule]:
        """Get CRA tax rules"""
        return {
            "Office Supplies": TaxRule(
                category="Business Expenses",
                deduction_type=TaxDeductionType.FULLY_DEDUCTIBLE,
                percentage=100.0,
                conditions=["Must be reasonable and for business use"],
                documentation=["Receipts"],
                notes="Fully deductible business expenses (T2125)",
            ),
            "Meals": TaxRule(
                category="Meals and Entertainment",
                deduction_type=TaxDeductionType.PARTIALLY_DEDUCTIBLE,
                percentage=50.0,
                conditions=["Must be for business purposes"],
                documentation=["Receipt", "Business purpose"],
                notes="50% deductible for business meals",
            ),
            "Vehicle": TaxRule(
                category="Motor Vehicle Expenses",
                deduction_type=TaxDeductionType.PARTIALLY_DEDUCTIBLE,
                percentage=None,
                conditions=["Must keep logbook", "Business use only"],
                documentation=["Vehicle logbook", "Receipts"],
                notes="Deductible based on business use percentage",
            ),
            "Home Office": TaxRule(
                category="Business-use-of-home",
                deduction_type=TaxDeductionType.CONDITIONAL,
                percentage=None,
                conditions=[
                    "Principal place of business OR",
                    "Used exclusively for business and meeting clients",
                ],
                documentation=["Square footage", "Home expenses"],
                notes="Based on percentage of home used for business (T2125)",
            ),
        }


class HMRCTaxRules:
    """HMRC (UK) tax rules"""

    @staticmethod
    def get_rules() -> Dict[str, TaxRule]:
        """Get HMRC tax rules"""
        return {
            "Office Supplies": TaxRule(
                category="Office Costs",
                deduction_type=TaxDeductionType.FULLY_DEDUCTIBLE,
                percentage=100.0,
                conditions=["Must be wholly and exclusively for business"],
                documentation=["Receipts"],
                notes="Allowable business expense",
            ),
            "Vehicle": TaxRule(
                category="Vehicle Expenses",
                deduction_type=TaxDeductionType.PARTIALLY_DEDUCTIBLE,
                percentage=None,
                conditions=["Business mileage only"],
                documentation=["Mileage log"],
                notes="Can use simplified expenses (45p/mile for first 10,000 miles)",
            ),
            "Home Office": TaxRule(
                category="Working from Home",
                deduction_type=TaxDeductionType.CONDITIONAL,
                percentage=None,
                conditions=["Must work from home regularly"],
                documentation=["May use flat rate or actual expenses"],
                notes="£6/week flat rate allowance or calculate actual business proportion",
            ),
        }


class TaxAssistant:
    """Main tax assistant for multi-region deduction analysis"""

    def __init__(self):
        self.rules = {
            "US": IRSTaxRules.get_rules(),
            "CA": CRATaxRules.get_rules(),
            "GB": HMRCTaxRules.get_rules(),
        }

    def _match_category_to_rule(
        self, description: str, category: Optional[str], region: str
    ) -> Optional[TaxRule]:
        """Match expense to tax rule"""
        region_rules = self.rules.get(region, {})

        # Direct category match
        if category:
            for rule_name, rule in region_rules.items():
                if rule_name.lower() in category.lower():
                    return rule

        # Keyword matching
        description_lower = description.lower()
        keywords = {
            "Office Supplies": ["office", "supplies", "staples", "paper", "pen"],
            "Software": ["software", "subscription", "saas", "app"],
            "Meals": ["restaurant", "meal", "lunch", "dinner", "food"],
            "Vehicle": ["gas", "fuel", "parking", "mileage", "car"],
            "Home Office": ["internet", "utilities", "rent", "mortgage"],
            "Travel": ["hotel", "flight", "airfare", "lodging"],
            "Medical": ["doctor", "hospital", "pharmacy", "medical"],
            "Education": ["course", "tuition", "training", "education"],
            "Advertising": ["advertising", "marketing", "ads"],
            "Professional Services": [
                "legal",
                "accounting",
                "consulting",
                "lawyer",
                "accountant",
            ],
        }

        for rule_name, rule_keywords in keywords.items():
            for keyword in rule_keywords:
                if keyword in description_lower:
                    if rule_name in region_rules:
                        return region_rules[rule_name]

        return None

    async def analyze_deduction(
        self,
        description: str,
        amount: float,
        category: Optional[str],
        region: str,
        tax_year: int,
        business_use: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Analyze expense for tax deductibility"""
        # Match to rule
        rule = self._match_category_to_rule(description, category, region)

        if not rule:
            # Default to not deductible
            return {
                "deduction_type": TaxDeductionType.NOT_DEDUCTIBLE,
                "deductible_amount": 0.0,
                "deductible_percentage": 0.0,
                "category": "Personal Expense",
                "conditions": [],
                "documentation_required": [],
                "notes": "No applicable tax deduction found. May be a personal expense.",
                "confidence": 0.5,
            }

        # Calculate deductible amount
        if rule.percentage is not None:
            deductible_percentage = rule.percentage
            deductible_amount = amount * (rule.percentage / 100.0)
        elif business_use is not None and business_use:
            # Assume 100% business use if specified
            deductible_percentage = 100.0
            deductible_amount = amount
        else:
            # Requires further analysis
            deductible_percentage = 0.0
            deductible_amount = 0.0

        # Region-specific categories
        irs_category = rule.category if region == "US" else None
        cra_category = rule.category if region == "CA" else None
        hmrc_category = rule.category if region == "GB" else None

        return {
            "deduction_type": rule.deduction_type,
            "deductible_amount": round(deductible_amount, 2),
            "deductible_percentage": deductible_percentage,
            "category": rule.category,
            "irs_category": irs_category,
            "cra_category": cra_category,
            "hmrc_category": hmrc_category,
            "conditions": rule.conditions,
            "documentation_required": rule.documentation,
            "notes": rule.notes,
            "confidence": 0.85,  # High confidence for rule-based matching
        }


# Global instance
tax_assistant = TaxAssistant()
