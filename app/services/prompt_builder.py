"""
System prompt builder for receipt extraction.
Engineered to produce accurate, structured JSON from OCR text.
"""

from typing import Optional


class PromptBuilder:
    """Builds system and user prompts for receipt extraction."""

    @staticmethod
    def get_extraction_system_prompt() -> str:
        """
        Get the system prompt for receipt extraction.

        This prompt is carefully engineered to:
        - Extract structured data from messy OCR text
        - Return only valid JSON (no markdown fences)
        - Handle missing/unclear data gracefully
        - Normalize merchant names
        - Infer item categories intelligently
        """
        return """You are a receipt data extraction specialist. Given OCR text from a receipt, extract structured information in JSON format.

Return ONLY valid JSON with this exact structure (no markdown code fences, no additional text):
{
  "merchant": {
    "name": "string (normalized name)",
    "store_number": "string or null",
    "address": "string or null"
  },
  "timestamp": "ISO8601 string or null",
  "items": [
    {
      "description": "string (item name)",
      "price": number,
      "quantity": number,
      "category": "string (inferred category like 'Groceries > Dairy')"
    }
  ],
  "totals": {
    "subtotal": number,
    "tax": number,
    "total": number
  },
  "payment_method": "string or null",
  "currency": "string (3-letter code like USD)"
}

CRITICAL RULES:
1. Return ONLY the JSON object - no markdown fences (```json), no explanations, no additional text
2. Normalize merchant names: "WAL*MART" → "Walmart", "TARGET T-1234" → "Target"
3. Infer intelligent categories: "MILK" → "Groceries > Dairy", "BREAD" → "Groceries > Bakery"
4. Use null for missing data (don't guess)
5. Parse dates into ISO8601 format when possible
6. Ensure all numbers are numeric types (not strings)
7. Handle OCR errors gracefully (e.g., "0" vs "O", "1" vs "l")
8. Default currency to "USD" unless specified
9. If subtotal/tax aren't listed, calculate from total if possible
10. Quantity defaults to 1 if not specified

EXAMPLES OF MERCHANT NORMALIZATION:
- "WAL*MART #1234" → "Walmart"
- "TARGET STORE T-0567" → "Target"
- "SAFEWAY #9876" → "Safeway"
- "CVS/PHARMACY" → "CVS Pharmacy"

EXAMPLES OF CATEGORY INFERENCE:
- "MILK", "EGGS", "CHEESE" → "Groceries > Dairy"
- "BREAD", "BAGELS", "MUFFINS" → "Groceries > Bakery"
- "APPLES", "BANANAS", "LETTUCE" → "Groceries > Produce"
- "SHAMPOO", "SOAP", "TOOTHPASTE" → "Health & Beauty"
- "IBUPROFEN", "VITAMINS" → "Health & Beauty > Medicine"
- "PEN", "PAPER", "NOTEBOOK" → "Office Supplies"
"""

    @staticmethod
    def build_extraction_user_prompt(ocr_text: str, image_url: Optional[str] = None) -> str:
        """
        Build the user prompt for receipt extraction.

        Args:
            ocr_text: The OCR text from the receipt
            image_url: Optional image URL for vision models

        Returns:
            The formatted user prompt
        """
        prompt = f"Extract receipt data from this OCR text:\n\n{ocr_text}"

        if image_url:
            prompt += f"\n\nImage URL: {image_url}"

        prompt += "\n\nRemember: Return ONLY the JSON object, no markdown fences or additional text."

        return prompt

    @staticmethod
    def build_chat_messages(ocr_text: str, image_url: Optional[str] = None) -> list[dict]:
        """
        Build chat messages for receipt extraction.

        Args:
            ocr_text: The OCR text from the receipt
            image_url: Optional image URL for vision models

        Returns:
            List of chat messages in OpenAI format
        """
        return [
            {
                "role": "system",
                "content": PromptBuilder.get_extraction_system_prompt()
            },
            {
                "role": "user",
                "content": PromptBuilder.build_extraction_user_prompt(ocr_text, image_url)
            }
        ]
