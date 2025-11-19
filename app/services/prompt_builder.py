"""System prompt builder for receipt extraction."""

from typing import Optional


class PromptBuilder:
    """Build system prompts for LLM extraction."""

    @staticmethod
    def get_extraction_system_prompt() -> str:
        """Get the system prompt for receipt extraction."""
        return """You are a receipt data extraction specialist. Given OCR text from a receipt, extract structured information in JSON format.

Return ONLY valid JSON with this exact structure (no markdown fences, no additional text):
{
  "merchant": {
    "name": "string",
    "store_number": "string or null",
    "address": "string or null",
    "phone": "string or null"
  },
  "timestamp": "ISO8601 string or null",
  "items": [
    {
      "description": "string",
      "price": number,
      "quantity": number,
      "category": "string or null"
    }
  ],
  "totals": {
    "subtotal": number,
    "tax": number,
    "total": number,
    "tip": number or null,
    "discount": number or null
  },
  "payment_method": "string or null",
  "currency": "string (default USD)"
}

Rules:
1. Normalize merchant names (e.g., "WAL*MART" → "Walmart", "AMZN.COM" → "Amazon")
2. Parse dates carefully and convert to ISO8601 format
3. Extract ALL items with prices
4. Calculate totals accurately
5. Infer item categories intelligently:
   - Food items → "Groceries > [Subcategory]"
   - Electronics → "Electronics > [Subcategory]"
   - Clothing → "Apparel > [Subcategory]"
   - Office supplies → "Office > [Subcategory]"
   - Services → "Services > [Type]"
6. Handle missing data gracefully (use null)
7. Ensure all numbers are numeric types (not strings)
8. NEVER include markdown code fences (```json) - return pure JSON only

Examples of category inference:
- "MILK 2% GAL" → "Groceries > Dairy"
- "BREAD WHEAT" → "Groceries > Bakery"
- "USB CABLE" → "Electronics > Accessories"
- "COFFEE LATTE" → "Food & Beverage > Coffee"
- "GAS FUEL" → "Transportation > Fuel"
"""

    @staticmethod
    def build_extraction_prompt(ocr_text: str, merchant_hint: Optional[str] = None) -> str:
        """Build user prompt for extraction."""
        prompt = f"Extract structured data from this receipt:\n\n{ocr_text}"
        if merchant_hint:
            prompt += f"\n\nMerchant hint: {merchant_hint}"
        return prompt

    @staticmethod
    def get_categorization_prompt(item_description: str, merchant: Optional[str] = None) -> str:
        """Build prompt for item categorization."""
        context = f" from {merchant}" if merchant else ""
        return f"""Categorize this item{context}: "{item_description}"

Return ONLY a JSON object with this structure:
{{
  "category": "Main Category > Subcategory",
  "confidence": 0.0-1.0,
  "tax_relevant": true/false,
  "business_expense_category": "IRS category or null"
}}

Categories should be hierarchical:
- Groceries > Dairy, Bakery, Produce, Meat, etc.
- Food & Beverage > Restaurant, Coffee, Fast Food, etc.
- Electronics > Computers, Accessories, Software, etc.
- Transportation > Fuel, Parking, Public Transit, etc.
- Office > Supplies, Furniture, Equipment, etc.
- Healthcare > Pharmacy, Medical, Dental, etc.
- Entertainment > Movies, Events, Streaming, etc.
- Utilities > Electric, Gas, Water, Internet, etc.
- Professional Services > Legal, Accounting, Consulting, etc.

For business_expense_category, use IRS categories:
- Advertising
- Car and truck expenses
- Depreciation
- Insurance
- Legal and professional services
- Office expense
- Rent or lease
- Supplies
- Travel
- Meals and entertainment (50% deductible)
- Utilities

Return pure JSON only (no markdown fences)."""

    @staticmethod
    def get_tax_analysis_prompt(
        merchant: str,
        items: list,
        total: float,
        jurisdiction: str,
    ) -> str:
        """Build prompt for tax deduction analysis."""
        items_text = "\n".join([f"- {item.get('description', 'Unknown')}: ${item.get('price', 0):.2f}" for item in items])

        jurisdiction_rules = {
            "US": """IRS Rules:
- Ordinary and necessary business expenses are deductible
- Meals: 50% deductible (100% if from restaurant in 2021-2022)
- Entertainment: Generally not deductible after TCJA
- Home office: Must be exclusive and regular use
- Mileage: $0.655/mile standard rate (2023)
- Record keeping: Must maintain receipts for expenses over $75""",
            "CA": """CRA Rules (Canada):
- Business expenses must be reasonable and incurred to earn income
- Meals and entertainment: 50% deductible
- Motor vehicle: Keep detailed log
- GST/HST: Can claim input tax credits
- Home office: Must be principal place of business or used regularly""",
            "GB": """HMRC Rules (UK):
- Wholly and exclusively for business purposes
- Subsistence: Reasonable meal costs when away on business
- Mileage: 45p per mile for first 10,000 miles
- VAT: Can reclaim on business purchases
- Must keep records for 5 years""",
        }

        return f"""Analyze this receipt for tax deductions ({jurisdiction} - {jurisdiction_rules.get(jurisdiction, 'US')}):

Merchant: {merchant}
Total Amount: ${total:.2f}

Items:
{items_text}

Provide a JSON response with this structure:
{{
  "deductible_amount": number,
  "deductible_percentage": number (0-100),
  "deductions": [
    {{
      "item": "string",
      "amount": number,
      "deductible": true/false,
      "percentage": number (0-100),
      "category": "IRS/CRA/HMRC category",
      "reasoning": "brief explanation"
    }}
  ],
  "recommendations": ["list of recommendations for maximizing deductions"],
  "warnings": ["list of potential issues or missing documentation"]
}}

Return pure JSON only (no markdown fences)."""
