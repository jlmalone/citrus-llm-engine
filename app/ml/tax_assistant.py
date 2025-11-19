"""Tax assistant for IRS/CRA/HMRC deduction identification."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.config import settings
from app.ml.categorization_engine import CategoryDatabase

logger = logging.getLogger(__name__)


class TaxRule:
    """Base class for tax rules."""

    def __init__(self, jurisdiction: str) -> None:
        self.jurisdiction = jurisdiction

    def evaluate(self, item: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if item is deductible."""
        raise NotImplementedError


class IRSTaxRules(TaxRule):
    """IRS (United States) tax rules."""

    def __init__(self) -> None:
        super().__init__("US")
        self.standard_mileage_rate = settings.irs_standard_mileage_rate

    def evaluate(self, item: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate IRS deductibility."""
        category = item.get("category", "")
        price = item.get("price", 0.0)
        description = item.get("description", "").lower()

        # Get category info
        main_category = category.split(">")[0].strip()
        category_info = CategoryDatabase.get_category_info(main_category)

        if not category_info:
            return self._not_deductible(item, "Category not recognized")

        # Check if category is tax deductible
        if not category_info.get("tax_deductible"):
            return self._not_deductible(item, f"{main_category} not deductible for business")

        irs_category = category_info.get("irs_category")

        # Special rules for specific categories
        if main_category == "Food & Beverage":
            return {
                "deductible": True,
                "percentage": 50.0,  # 50% deductible for meals
                "category": "Meals and entertainment",
                "reasoning": "Business meals are 50% deductible under IRS rules",
                "notes": "Must have business purpose documented",
            }

        elif main_category == "Healthcare":
            # Medical expenses only if > 7.5% AGI
            return {
                "deductible": True,
                "percentage": 100.0,
                "category": "Medical expenses",
                "reasoning": "Deductible if total medical expenses exceed 7.5% of AGI",
                "notes": "Track all medical expenses throughout the year",
            }

        elif main_category == "Electronics":
            if price > 2500:
                return {
                    "deductible": True,
                    "percentage": 100.0,
                    "category": "Depreciation (Section 179)",
                    "reasoning": "Equipment over $2,500 must be depreciated",
                    "notes": "Consider Section 179 immediate expensing up to $1,160,000",
                }

        elif main_category == "Transportation":
            if "fuel" in description or "gas" in description:
                return {
                    "deductible": True,
                    "percentage": 100.0,
                    "category": "Car and truck expenses",
                    "reasoning": "Fuel for business use is fully deductible",
                    "notes": f"Or use standard mileage rate: ${self.standard_mileage_rate}/mile",
                }

        # Default: fully deductible if category allows
        return {
            "deductible": True,
            "percentage": 100.0,
            "category": irs_category or "Business expense",
            "reasoning": f"{main_category} is an ordinary and necessary business expense",
            "notes": "Keep receipt and document business purpose",
        }

    def _not_deductible(self, item: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Return not deductible result."""
        return {
            "deductible": False,
            "percentage": 0.0,
            "category": "Not deductible",
            "reasoning": reason,
            "notes": "Personal expense",
        }


class CRATaxRules(TaxRule):
    """CRA (Canada) tax rules."""

    def __init__(self) -> None:
        super().__init__("CA")
        self.gst_hst_rate = settings.cra_gst_hst_rate

    def evaluate(self, item: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate CRA deductibility."""
        category = item.get("category", "")
        price = item.get("price", 0.0)
        description = item.get("description", "").lower()

        main_category = category.split(">")[0].strip()
        category_info = CategoryDatabase.get_category_info(main_category)

        if not category_info or not category_info.get("tax_deductible"):
            return {
                "deductible": False,
                "percentage": 0.0,
                "category": "Not deductible",
                "reasoning": "Not an eligible business expense",
                "notes": "Personal expense",
            }

        cra_category = category_info.get("cra_category")

        # Special rules
        if main_category == "Food & Beverage":
            return {
                "deductible": True,
                "percentage": 50.0,
                "category": "Meals and entertainment",
                "reasoning": "Meals and entertainment are 50% deductible",
                "notes": "GST/HST can be claimed on full amount",
            }

        elif main_category == "Transportation":
            return {
                "deductible": True,
                "percentage": 100.0,
                "category": "Motor vehicle expenses",
                "reasoning": "Vehicle expenses for business use are deductible",
                "notes": "Maintain detailed vehicle log for CRA",
            }

        return {
            "deductible": True,
            "percentage": 100.0,
            "category": cra_category or "Business expense",
            "reasoning": f"{main_category} incurred to earn business income",
            "notes": "Expense must be reasonable and documented",
        }


class HMRCTaxRules(TaxRule):
    """HMRC (United Kingdom) tax rules."""

    def __init__(self) -> None:
        super().__init__("GB")
        self.vat_standard_rate = settings.hmrc_vat_standard_rate

    def evaluate(self, item: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate HMRC deductibility."""
        category = item.get("category", "")
        price = item.get("price", 0.0)
        description = item.get("description", "").lower()

        main_category = category.split(">")[0].strip()
        category_info = CategoryDatabase.get_category_info(main_category)

        if not category_info or not category_info.get("tax_deductible"):
            return {
                "deductible": False,
                "percentage": 0.0,
                "category": "Not allowable",
                "reasoning": "Not wholly and exclusively for business",
                "notes": "Disallowed expense",
            }

        hmrc_category = category_info.get("hmrc_category")

        # Special rules
        if main_category == "Food & Beverage":
            return {
                "deductible": True,
                "percentage": 100.0,
                "category": "Subsistence",
                "reasoning": "Reasonable meal costs while away on business",
                "notes": "Must be necessary for business travel",
            }

        elif main_category == "Transportation":
            if "fuel" in description:
                return {
                    "deductible": True,
                    "percentage": 100.0,
                    "category": "Travel - mileage",
                    "reasoning": "Business mileage at 45p per mile (first 10,000 miles)",
                    "notes": "Maintain mileage log for HMRC",
                }

        elif main_category == "Healthcare":
            return {
                "deductible": False,
                "percentage": 0.0,
                "category": "Not allowable",
                "reasoning": "Medical expenses not deductible for self-employed",
                "notes": "Use NHS or personal funds",
            }

        return {
            "deductible": True,
            "percentage": 100.0,
            "category": hmrc_category or "Business expense",
            "reasoning": f"{main_category} wholly and exclusively for business",
            "notes": "VAT reclaimable if VAT registered",
        }


class TaxAssistant:
    """Tax assistant for analyzing receipts across jurisdictions."""

    def __init__(self) -> None:
        """Initialize tax assistant."""
        self.rules = {
            "US": IRSTaxRules(),
            "CA": CRATaxRules(),
            "GB": HMRCTaxRules(),
        }
        logger.info("Tax assistant initialized")

    def analyze_receipt(
        self,
        merchant: str,
        items: List[Dict[str, Any]],
        total_amount: float,
        jurisdiction: str = "US",
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze receipt for tax deductions."""
        if jurisdiction not in self.rules:
            raise ValueError(f"Unsupported jurisdiction: {jurisdiction}")

        tax_rule = self.rules[jurisdiction]

        context = {
            "merchant": merchant,
            "total_amount": total_amount,
            "date": date,
            "jurisdiction": jurisdiction,
        }

        deductions = []
        deductible_total = 0.0

        for item in items:
            evaluation = tax_rule.evaluate(item, context)

            deductible_amount = item.get("price", 0.0) * (evaluation["percentage"] / 100.0)
            deductible_total += deductible_amount

            deductions.append({
                "item": item.get("description", "Unknown"),
                "amount": item.get("price", 0.0),
                "deductible": evaluation["deductible"],
                "deductible_amount": round(deductible_amount, 2),
                "percentage": evaluation["percentage"],
                "category": evaluation["category"],
                "reasoning": evaluation["reasoning"],
                "notes": evaluation.get("notes", ""),
            })

        # Generate recommendations
        recommendations = self._generate_recommendations(
            deductions, total_amount, jurisdiction, merchant
        )

        # Generate warnings
        warnings = self._generate_warnings(deductions, total_amount, jurisdiction)

        return {
            "jurisdiction": jurisdiction,
            "total_amount": round(total_amount, 2),
            "deductible_amount": round(deductible_total, 2),
            "deductible_percentage": round((deductible_total / total_amount * 100) if total_amount > 0 else 0, 2),
            "deductions": deductions,
            "recommendations": recommendations,
            "warnings": warnings,
        }

    def _generate_recommendations(
        self,
        deductions: List[Dict[str, Any]],
        total: float,
        jurisdiction: str,
        merchant: str,
    ) -> List[str]:
        """Generate tax optimization recommendations."""
        recommendations = []

        # Check for meal expenses
        meal_items = [d for d in deductions if "Meals" in d.get("category", "")]
        if meal_items:
            recommendations.append(
                "Document business purpose of meals (e.g., client meeting, business travel)"
            )

        # Check for large purchases
        large_items = [d for d in deductions if d.get("amount", 0) > 500]
        if large_items and jurisdiction == "US":
            recommendations.append(
                "Consider Section 179 deduction for equipment purchases over $2,500"
            )

        # Vehicle expenses
        vehicle_items = [d for d in deductions if "vehicle" in d.get("category", "").lower() or "mileage" in d.get("category", "").lower()]
        if vehicle_items:
            recommendations.append(
                f"Maintain detailed mileage log for {jurisdiction} tax authority"
            )

        # General recommendation
        if total > 75 and jurisdiction == "US":
            recommendations.append(
                "IRS requires receipts for expenses over $75 - this receipt meets that threshold"
            )

        return recommendations

    def _generate_warnings(
        self,
        deductions: List[Dict[str, Any]],
        total: float,
        jurisdiction: str,
    ) -> List[str]:
        """Generate warnings about potential issues."""
        warnings = []

        # Check for non-deductible items
        non_deductible = [d for d in deductions if not d.get("deductible")]
        if non_deductible:
            warnings.append(
                f"{len(non_deductible)} item(s) may not be deductible - ensure proper business purpose"
            )

        # Check for partial deductions
        partial = [d for d in deductions if 0 < d.get("percentage", 100) < 100]
        if partial:
            warnings.append(
                f"{len(partial)} item(s) are only partially deductible - verify percentage rules"
            )

        return warnings
