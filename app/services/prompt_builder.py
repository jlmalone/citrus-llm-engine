"""
Prompt builder for LLM interactions
"""
from typing import Dict, List, Optional
from datetime import datetime


class PromptBuilder:
    """Build prompts for various LLM tasks"""

    @staticmethod
    def build_extraction_prompt(ocr_text: str, language: str = "en") -> List[Dict[str, str]]:
        """Build prompt for receipt extraction"""
        system_prompt = """You are a receipt data extraction specialist. Given OCR text from a receipt, extract structured information in JSON format.

Return ONLY valid JSON with this exact structure (no markdown fences, no additional text):
{
  "merchant": {
    "name": "string - normalized merchant name",
    "store_number": "string or null",
    "address": "string or null",
    "phone": "string or null",
    "url": "string or null"
  },
  "timestamp": "ISO8601 datetime string or null",
  "items": [
    {
      "description": "string - normalized item description",
      "price": number,
      "quantity": number,
      "category": "string - inferred category",
      "unit_price": number or null,
      "discount": number or null
    }
  ],
  "totals": {
    "subtotal": number,
    "tax": number,
    "total": number,
    "tip": number or null,
    "discount": number or null,
    "cashback": number or null
  },
  "payment_method": "string or null (e.g., 'VISA-1234', 'CASH', 'DEBIT')",
  "currency": "string - 3-letter currency code (default 'USD')",
  "receipt_number": "string or null",
  "tax_id": "string or null"
}

Rules:
1. Normalize merchant names (WAL*MART → Walmart, MCDONALDS → McDonald's)
2. Infer item categories intelligently (MILK → Groceries > Dairy, GAS → Transportation > Fuel)
3. Parse dates to ISO8601 format
4. Handle missing data gracefully (use null)
5. Clean up item descriptions (remove extra spaces, normalize case)
6. Calculate unit_price if quantity > 1
7. Return ONLY the JSON object, no other text"""

        user_prompt = f"""Extract structured data from this receipt:

{ocr_text}

Return the JSON object now:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def build_categorization_prompt(
        description: str,
        merchant: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for expense categorization"""
        system_prompt = """You are an expense categorization expert. Categorize expenses into hierarchical categories.

Return ONLY valid JSON with this structure:
{
  "category": "Primary > Secondary > Tertiary",
  "confidence": 0.95,
  "tax_category": "string - tax deduction category",
  "deductible": boolean,
  "subcategories": ["alt1", "alt2"]
}

Standard categories:
- Transportation > Fuel, Parking, Public Transit, Rideshare, Maintenance
- Groceries > Dairy, Meat, Produce, Bakery, Household
- Dining > Restaurants, Fast Food, Coffee Shops, Bars
- Entertainment > Movies, Concerts, Sports, Streaming
- Healthcare > Pharmacy, Doctor, Dental, Vision, Hospital
- Shopping > Clothing, Electronics, Home Goods, Personal Care
- Utilities > Electric, Gas, Water, Internet, Phone
- Business > Office Supplies, Software, Equipment, Professional Services
- Travel > Flights, Hotels, Car Rental, Tours
- Education > Tuition, Books, Courses, Supplies

Return ONLY the JSON object."""

        context_parts = [f"Description: {description}"]
        if merchant:
            context_parts.append(f"Merchant: {merchant}")
        if amount:
            context_parts.append(f"Amount: ${amount:.2f}")

        user_prompt = f"""Categorize this expense:

{chr(10).join(context_parts)}

Return the JSON object now:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def build_tax_assistant_prompt(
        description: str,
        amount: float,
        category: Optional[str],
        region: str,
        tax_year: int,
        business_use: Optional[bool] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt for tax deduction analysis"""
        region_info = {
            "US": "IRS (United States) tax regulations",
            "CA": "CRA (Canada Revenue Agency) tax regulations",
            "GB": "HMRC (United Kingdom) tax regulations",
        }

        system_prompt = f"""You are a tax assistant expert for {region_info.get(region, region)} for tax year {tax_year}.

Analyze expenses and determine tax deductibility with detailed reasoning.

Return ONLY valid JSON with this structure:
{{
  "deduction_type": "fully_deductible" | "partially_deductible" | "not_deductible" | "requires_documentation" | "conditional",
  "deductible_amount": number,
  "deductible_percentage": number (0-100),
  "category": "string - tax category",
  "irs_category": "string or null (for US)",
  "cra_category": "string or null (for Canada)",
  "hmrc_category": "string or null (for UK)",
  "conditions": ["list of conditions that must be met"],
  "documentation_required": ["list of required documentation"],
  "notes": "string - detailed explanation"
}}

Key considerations for {region}:
- Business vs personal use percentage
- Ordinary and necessary business expense test
- Documentation requirements
- Limitations and caps
- Special rules for specific categories

Return ONLY the JSON object."""

        context_parts = [
            f"Description: {description}",
            f"Amount: ${amount:.2f}",
            f"Region: {region}",
            f"Tax Year: {tax_year}",
        ]
        if category:
            context_parts.append(f"Category: {category}")
        if business_use is not None:
            context_parts.append(f"Business Use: {'Yes' if business_use else 'No'}")

        user_prompt = f"""Analyze this expense for tax deductibility:

{chr(10).join(context_parts)}

Return the JSON object now:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def build_custom_prompt(
        system_message: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build custom prompt with conversation history"""
        messages = [{"role": "system", "content": system_message}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        return messages


# Global instance
prompt_builder = PromptBuilder()
