"""ML-based categorization engine with tax mapping."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class CategoryDatabase:
    """Database of categories with tax mappings."""

    # Hierarchical category structure with tax implications
    CATEGORIES = {
        "Groceries": {
            "subcategories": ["Dairy", "Bakery", "Produce", "Meat", "Beverages", "Snacks"],
            "tax_deductible": False,
            "irs_category": None,
            "cra_category": None,
            "hmrc_category": None,
            "keywords": ["milk", "bread", "eggs", "cheese", "fruit", "vegetable", "meat", "juice"],
        },
        "Food & Beverage": {
            "subcategories": ["Restaurant", "Coffee", "Fast Food", "Bar"],
            "tax_deductible": True,
            "irs_category": "Meals (50% deductible)",
            "cra_category": "Meals and entertainment (50%)",
            "hmrc_category": "Subsistence",
            "keywords": ["restaurant", "coffee", "cafe", "lunch", "dinner", "breakfast", "starbucks"],
        },
        "Transportation": {
            "subcategories": ["Fuel", "Parking", "Public Transit", "Tolls", "Rideshare"],
            "tax_deductible": True,
            "irs_category": "Car and truck expenses",
            "cra_category": "Motor vehicle expenses",
            "hmrc_category": "Travel - mileage",
            "keywords": ["gas", "fuel", "parking", "uber", "lyft", "taxi", "metro", "toll"],
        },
        "Office": {
            "subcategories": ["Supplies", "Furniture", "Equipment", "Software"],
            "tax_deductible": True,
            "irs_category": "Office expense",
            "cra_category": "Office expenses",
            "hmrc_category": "Office costs",
            "keywords": ["paper", "pen", "desk", "chair", "computer", "printer", "stapler"],
        },
        "Electronics": {
            "subcategories": ["Computers", "Accessories", "Software", "Mobile"],
            "tax_deductible": True,
            "irs_category": "Depreciation (if > $2500)",
            "cra_category": "Capital cost allowance",
            "hmrc_category": "Capital allowances",
            "keywords": ["laptop", "phone", "tablet", "cable", "charger", "mouse", "keyboard"],
        },
        "Utilities": {
            "subcategories": ["Electric", "Gas", "Water", "Internet", "Phone"],
            "tax_deductible": True,
            "irs_category": "Utilities",
            "cra_category": "Utilities",
            "hmrc_category": "Use of home",
            "keywords": ["electricity", "water", "internet", "wifi", "phone", "cellular"],
        },
        "Professional Services": {
            "subcategories": ["Legal", "Accounting", "Consulting", "Marketing"],
            "tax_deductible": True,
            "irs_category": "Legal and professional services",
            "cra_category": "Professional fees",
            "hmrc_category": "Professional fees",
            "keywords": ["lawyer", "accountant", "consultant", "advisor", "attorney"],
        },
        "Healthcare": {
            "subcategories": ["Pharmacy", "Medical", "Dental", "Vision"],
            "tax_deductible": True,
            "irs_category": "Medical (if > 7.5% AGI)",
            "cra_category": "Medical expenses",
            "hmrc_category": "Not deductible",
            "keywords": ["pharmacy", "doctor", "dentist", "medicine", "prescription", "hospital"],
        },
        "Entertainment": {
            "subcategories": ["Movies", "Events", "Streaming", "Sports"],
            "tax_deductible": False,
            "irs_category": "Not deductible (post-TCJA)",
            "cra_category": "Not deductible",
            "hmrc_category": "Not deductible",
            "keywords": ["movie", "concert", "netflix", "spotify", "game", "ticket"],
        },
        "Travel": {
            "subcategories": ["Airfare", "Hotel", "Rental Car", "Meals"],
            "tax_deductible": True,
            "irs_category": "Travel",
            "cra_category": "Travel",
            "hmrc_category": "Travel - subsistence",
            "keywords": ["flight", "hotel", "airbnb", "rental", "airline", "airport"],
        },
        "Insurance": {
            "subcategories": ["Business", "Health", "Liability", "Property"],
            "tax_deductible": True,
            "irs_category": "Insurance",
            "cra_category": "Insurance",
            "hmrc_category": "Insurance",
            "keywords": ["insurance", "policy", "premium", "coverage"],
        },
        "Advertising": {
            "subcategories": ["Online Ads", "Print", "Social Media", "SEO"],
            "tax_deductible": True,
            "irs_category": "Advertising",
            "cra_category": "Advertising",
            "hmrc_category": "Advertising",
            "keywords": ["google ads", "facebook ads", "marketing", "advertising", "promotion"],
        },
        "Education": {
            "subcategories": ["Courses", "Books", "Training", "Conferences"],
            "tax_deductible": True,
            "irs_category": "Education (if work-related)",
            "cra_category": "Training",
            "hmrc_category": "Training",
            "keywords": ["course", "training", "seminar", "conference", "book", "education"],
        },
    }

    @classmethod
    def get_all_keywords(cls) -> Dict[str, str]:
        """Get all keywords mapped to categories."""
        keyword_map = {}
        for category, data in cls.CATEGORIES.items():
            for keyword in data["keywords"]:
                keyword_map[keyword.lower()] = category
        return keyword_map

    @classmethod
    def get_category_info(cls, category: str) -> Optional[Dict[str, Any]]:
        """Get information about a category."""
        return cls.CATEGORIES.get(category)


class CategorizationEngine:
    """ML-based categorization engine using embeddings and keyword matching."""

    def __init__(self) -> None:
        """Initialize categorization engine."""
        self.model = None
        self.keyword_map = CategoryDatabase.get_all_keywords()
        logger.info("Categorization engine initialized")

    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.categorization_model)
        return self.model

    def _keyword_match(self, text: str) -> Optional[Tuple[str, float]]:
        """Match using keyword database."""
        text_lower = text.lower()
        for keyword, category in self.keyword_map.items():
            if keyword in text_lower:
                # Calculate confidence based on keyword length and position
                confidence = min(0.95, 0.7 + (len(keyword) / len(text)) * 0.25)
                return category, confidence
        return None

    def _embedding_match(self, text: str) -> Tuple[str, float]:
        """Match using semantic embeddings."""
        try:
            model = self._get_model()

            # Get text embedding
            text_embedding = model.encode([text])[0]

            # Get category embeddings
            categories = list(CategoryDatabase.CATEGORIES.keys())
            category_embeddings = model.encode(categories)

            # Calculate similarities
            similarities = np.dot(category_embeddings, text_embedding) / (
                np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(text_embedding)
            )

            # Get best match
            best_idx = np.argmax(similarities)
            best_category = categories[best_idx]
            confidence = float(similarities[best_idx])

            return best_category, confidence

        except Exception as e:
            logger.error(f"Embedding match failed: {e}")
            return "Uncategorized", 0.5

    def categorize(
        self,
        item_description: str,
        merchant: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Categorize an item description."""
        # Try keyword matching first (fast and accurate)
        keyword_result = self._keyword_match(item_description)

        if keyword_result and keyword_result[1] >= settings.categorization_confidence_threshold:
            category, confidence = keyword_result
        else:
            # Fall back to embedding matching
            category, confidence = self._embedding_match(item_description)

        # Get category info
        category_info = CategoryDatabase.get_category_info(category)

        if not category_info:
            return {
                "category": "Uncategorized",
                "confidence": 0.5,
                "tax_deductible": False,
                "irs_category": None,
                "cra_category": None,
                "hmrc_category": None,
            }

        # Infer subcategory based on keywords
        subcategory = None
        for sub in category_info["subcategories"]:
            if sub.lower() in item_description.lower():
                subcategory = sub
                break

        full_category = f"{category} > {subcategory}" if subcategory else category

        return {
            "category": full_category,
            "confidence": confidence,
            "tax_deductible": category_info["tax_deductible"],
            "irs_category": category_info["irs_category"],
            "cra_category": category_info["cra_category"],
            "hmrc_category": category_info["hmrc_category"],
        }

    def categorize_batch(
        self,
        items: List[str],
        merchant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Categorize multiple items efficiently."""
        return [self.categorize(item, merchant) for item in items]

    def get_tax_category(
        self,
        category: str,
        jurisdiction: str = "US",
    ) -> Optional[str]:
        """Get tax category for a given jurisdiction."""
        # Extract main category
        main_category = category.split(">")[0].strip()
        category_info = CategoryDatabase.get_category_info(main_category)

        if not category_info:
            return None

        jurisdiction_map = {
            "US": "irs_category",
            "CA": "cra_category",
            "GB": "hmrc_category",
        }

        field = jurisdiction_map.get(jurisdiction)
        if field:
            return category_info.get(field)

        return None
