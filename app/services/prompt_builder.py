"""System prompt management for receipt extraction."""

from typing import Dict, Any


RECEIPT_EXTRACTION_SYSTEM_PROMPT = """You are a receipt data extraction specialist. Given OCR text from a receipt, extract structured information in JSON format.

Return ONLY valid JSON with this exact structure (no markdown fences, no additional text):
{
  "merchant": {
    "name": "string - normalized merchant name",
    "store_number": "string or null",
    "address": "string or null"
  },
  "timestamp": "ISO8601 string or null (e.g., 2024-12-25T15:45:00Z)",
  "items": [
    {
      "description": "string - item description",
      "price": number,
      "quantity": integer,
      "category": "string - inferred category (e.g., 'Groceries > Dairy')"
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
  "currency": "string - 3-letter currency code (default: USD)"
}

CRITICAL RULES:
1. Normalize merchant names (e.g., "WAL*MART" → "Walmart", "TARGET #1234" → "Target")
2. Infer intelligent categories for items:
   - MILK, CHEESE, YOGURT → "Groceries > Dairy"
   - BREAD, BAGELS → "Groceries > Bakery"
   - CHICKEN, BEEF → "Groceries > Meat"
   - CHIPS, CANDY → "Groceries > Snacks"
   - SHAMPOO, SOAP → "Health & Beauty"
   - Use your best judgment for other items
3. Handle missing data gracefully - use null for unavailable fields
4. Parse dates/times into ISO8601 format when possible
5. Extract ALL numeric values as numbers, not strings
6. Default quantity to 1 if not specified
7. NEVER include markdown code fences (```json) in your response
8. Return ONLY the JSON object, nothing else
9. If OCR text is unclear or ambiguous, make your best inference
10. Ensure all monetary values are precise decimals (e.g., 3.99, not 4)

Example input:
"WALMART SUPERCENTER #1234
123 MAIN ST
12/25/2024 3:45 PM

MILK 2% GAL    3.99
BREAD WHEAT    2.49
EGGS DOZEN     4.29

SUBTOTAL      10.77
TAX            0.75
TOTAL         11.52

VISA ****1234"

Example output:
{
  "merchant": {
    "name": "Walmart",
    "store_number": "1234",
    "address": "123 Main St"
  },
  "timestamp": "2024-12-25T15:45:00Z",
  "items": [
    {
      "description": "Milk 2% Gallon",
      "price": 3.99,
      "quantity": 1,
      "category": "Groceries > Dairy"
    },
    {
      "description": "Bread Wheat",
      "price": 2.49,
      "quantity": 1,
      "category": "Groceries > Bakery"
    },
    {
      "description": "Eggs Dozen",
      "price": 4.29,
      "quantity": 1,
      "category": "Groceries > Dairy"
    }
  ],
  "totals": {
    "subtotal": 10.77,
    "tax": 0.75,
    "total": 11.52,
    "tip": null,
    "discount": null
  },
  "payment_method": "Visa",
  "currency": "USD"
}

Now extract the receipt data from the OCR text provided."""


class PromptBuilder:
    """Builds prompts for LLM interactions."""

    @staticmethod
    def build_extraction_prompt(ocr_text: str) -> str:
        """Build a prompt for receipt extraction.

        Args:
            ocr_text: OCR text from the receipt

        Returns:
            Formatted prompt for the LLM
        """
        return f"""Receipt OCR Text:
{ocr_text}

Extract the structured data from this receipt following the format and rules specified."""

    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for receipt extraction.

        Returns:
            System prompt string
        """
        return RECEIPT_EXTRACTION_SYSTEM_PROMPT

    @staticmethod
    def build_messages(ocr_text: str) -> list[Dict[str, Any]]:
        """Build OpenAI-compatible message array for extraction.

        Args:
            ocr_text: OCR text from the receipt

        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": RECEIPT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": PromptBuilder.build_extraction_prompt(ocr_text)},
        ]
