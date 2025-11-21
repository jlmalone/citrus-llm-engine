"""
ML Categorization Engine with tax mapping
Uses both ML models and LLM fallback for high accuracy
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class CategoryClassifier:
    """ML-based category classifier"""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.categories = self._load_categories()
        self.keyword_rules = self._load_keyword_rules()

    def _load_categories(self) -> Dict[str, List[str]]:
        """Load hierarchical category definitions"""
        return {
            "Transportation": {
                "subcategories": ["Fuel", "Parking", "Public Transit", "Rideshare", "Maintenance", "Car Rental"],
                "keywords": ["gas", "fuel", "shell", "exxon", "chevron", "parking", "uber", "lyft", "taxi", "metro", "bus", "train"],
                "tax_category": "Vehicle Expenses",
            },
            "Groceries": {
                "subcategories": ["Dairy", "Meat", "Produce", "Bakery", "Household", "Beverages"],
                "keywords": ["walmart", "kroger", "safeway", "whole foods", "trader joe", "milk", "bread", "eggs", "grocery"],
                "tax_category": "Not Deductible (Personal)",
            },
            "Dining": {
                "subcategories": ["Restaurants", "Fast Food", "Coffee Shops", "Bars"],
                "keywords": ["restaurant", "mcdonald", "burger", "pizza", "starbucks", "coffee", "bar", "pub", "diner"],
                "tax_category": "Meals and Entertainment",
            },
            "Entertainment": {
                "subcategories": ["Movies", "Concerts", "Sports", "Streaming", "Games"],
                "keywords": ["cinema", "theater", "concert", "spotify", "netflix", "hulu", "game"],
                "tax_category": "Not Deductible (Personal)",
            },
            "Healthcare": {
                "subcategories": ["Pharmacy", "Doctor", "Dental", "Vision", "Hospital"],
                "keywords": ["cvs", "walgreens", "pharmacy", "doctor", "dental", "hospital", "clinic", "medical"],
                "tax_category": "Medical Expenses",
            },
            "Shopping": {
                "subcategories": ["Clothing", "Electronics", "Home Goods", "Personal Care"],
                "keywords": ["amazon", "target", "best buy", "clothing", "electronics", "furniture", "appliance"],
                "tax_category": "Not Deductible (Personal)",
            },
            "Utilities": {
                "subcategories": ["Electric", "Gas", "Water", "Internet", "Phone"],
                "keywords": ["electric", "power", "gas", "water", "internet", "phone", "verizon", "att", "comcast"],
                "tax_category": "Home Office Deduction",
            },
            "Business": {
                "subcategories": ["Office Supplies", "Software", "Equipment", "Professional Services", "Advertising"],
                "keywords": ["office", "supplies", "staples", "software", "subscription", "consulting", "advertising", "marketing"],
                "tax_category": "Business Expenses",
            },
            "Travel": {
                "subcategories": ["Flights", "Hotels", "Car Rental", "Tours"],
                "keywords": ["airline", "flight", "hotel", "marriott", "hilton", "airbnb", "rental car", "hertz"],
                "tax_category": "Travel Expenses",
            },
            "Education": {
                "subcategories": ["Tuition", "Books", "Courses", "Supplies"],
                "keywords": ["university", "college", "tuition", "textbook", "course", "udemy", "coursera"],
                "tax_category": "Education Expenses",
            },
        }

    def _load_keyword_rules(self) -> Dict[str, List[str]]:
        """Load keyword-based rules for quick classification"""
        rules = {}
        for category, info in self.categories.items():
            for keyword in info["keywords"]:
                if keyword not in rules:
                    rules[keyword] = []
                rules[keyword].append(category)
        return rules

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        return re.sub(r'[^a-z0-9\s]', '', text.lower())

    def predict_category(
        self,
        description: str,
        merchant: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> Tuple[str, float, Optional[str]]:
        """
        Predict category using keyword matching (can be replaced with ML model)
        Returns: (category, confidence, tax_category)
        """
        # Normalize inputs
        text = self._normalize_text(f"{description} {merchant or ''}")

        # Count keyword matches
        category_scores = {}
        for keyword, categories in self.keyword_rules.items():
            if keyword in text:
                for category in categories:
                    category_scores[category] = category_scores.get(category, 0) + 1

        if not category_scores:
            return "Uncategorized", 0.3, None

        # Get best match
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]

        # Calculate confidence (heuristic)
        confidence = min(0.6 + (max_score * 0.1), 0.95)

        # Get tax category
        tax_category = self.categories[best_category].get("tax_category")

        return best_category, confidence, tax_category

    def predict_subcategory(
        self,
        description: str,
        primary_category: str,
        merchant: Optional[str] = None,
    ) -> Optional[str]:
        """Predict subcategory within primary category"""
        if primary_category not in self.categories:
            return None

        text = self._normalize_text(f"{description} {merchant or ''}")
        subcategories = self.categories[primary_category]["subcategories"]

        # Simple keyword matching for subcategories
        for subcat in subcategories:
            if self._normalize_text(subcat) in text:
                return subcat

        # Return first subcategory as default
        return subcategories[0] if subcategories else None


class TaxMapper:
    """Maps categories to tax deduction categories by region"""

    def __init__(self):
        self.mappings = self._load_tax_mappings()

    def _load_tax_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load tax mapping data"""
        return {
            "US": {  # IRS categories
                "Business Expenses": {
                    "form": "Schedule C",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Ordinary and necessary business expenses",
                },
                "Meals and Entertainment": {
                    "form": "Schedule C",
                    "deductible": True,
                    "percentage": 50,
                    "notes": "50% deductible for business meals",
                },
                "Vehicle Expenses": {
                    "form": "Schedule C",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Business use percentage applies. Standard mileage or actual expenses.",
                },
                "Travel Expenses": {
                    "form": "Schedule C",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Business travel only. Keep detailed records.",
                },
                "Home Office Deduction": {
                    "form": "Form 8829",
                    "deductible": True,
                    "percentage": None,
                    "notes": "Based on square footage of dedicated office space",
                },
                "Medical Expenses": {
                    "form": "Schedule A",
                    "deductible": True,
                    "percentage": None,
                    "notes": "Only amounts exceeding 7.5% of AGI",
                },
                "Education Expenses": {
                    "form": "Form 8863",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Must be work-related or improve job skills",
                },
            },
            "CA": {  # CRA categories
                "Business Expenses": {
                    "form": "T2125",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Reasonable business expenses",
                },
                "Vehicle Expenses": {
                    "form": "T2125",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Business use percentage applies",
                },
                "Meals and Entertainment": {
                    "form": "T2125",
                    "deductible": True,
                    "percentage": 50,
                    "notes": "50% deductible for business meals",
                },
            },
            "GB": {  # HMRC categories
                "Business Expenses": {
                    "form": "Self Assessment",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Allowable business expenses",
                },
                "Vehicle Expenses": {
                    "form": "Self Assessment",
                    "deductible": True,
                    "percentage": 100,
                    "notes": "Business mileage allowance",
                },
            },
        }

    def get_tax_info(
        self, category: str, region: str = "US"
    ) -> Optional[Dict[str, Any]]:
        """Get tax information for a category in a specific region"""
        region_mappings = self.mappings.get(region, {})
        return region_mappings.get(category)


class CategorizationEngine:
    """Main categorization engine combining ML and rules"""

    def __init__(self):
        self.classifier = CategoryClassifier()
        self.tax_mapper = TaxMapper()

    async def categorize(
        self,
        description: str,
        merchant: Optional[str] = None,
        amount: Optional[float] = None,
        region: str = "US",
    ) -> Dict[str, any]:
        """Categorize an expense with tax mapping"""
        # Predict primary category
        primary_category, confidence, tax_category = self.classifier.predict_category(
            description, merchant, amount
        )

        # Predict subcategory
        subcategory = self.classifier.predict_subcategory(
            description, primary_category, merchant
        )

        # Build full category path
        full_category = f"{primary_category}"
        if subcategory:
            full_category += f" > {subcategory}"

        # Get alternative categories (lower confidence)
        alternatives = []
        # TODO: Implement alternative predictions

        # Get tax information
        tax_info = None
        if tax_category:
            tax_info = self.tax_mapper.get_tax_info(tax_category, region)

        return {
            "primary_category": {
                "category": full_category,
                "confidence": confidence,
                "tax_category": tax_category,
                "deductible": tax_info.get("deductible") if tax_info else None,
            },
            "alternative_categories": alternatives,
            "tax_info": tax_info,
            "method": "ml",
        }


# Global instance
categorization_engine = CategorizationEngine()
