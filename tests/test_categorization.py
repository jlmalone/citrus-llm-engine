"""
Tests for categorization engine
"""
import pytest


class TestCategorizationEngine:
    """Test categorization engine"""

    def test_categorize_groceries(self):
        """Test grocery categorization"""
        from app.ml.categorization_engine import CategoryClassifier

        classifier = CategoryClassifier()
        category, confidence, tax_category = classifier.predict_category(
            description="Milk and bread",
            merchant="Walmart"
        )

        assert "Groceries" in category or "Shopping" in category
        assert confidence > 0

    def test_categorize_gas(self):
        """Test fuel categorization"""
        from app.ml.categorization_engine import CategoryClassifier

        classifier = CategoryClassifier()
        category, confidence, tax_category = classifier.predict_category(
            description="Gas",
            merchant="Shell"
        )

        assert "Transportation" in category
        assert confidence > 0.5

    def test_categorize_restaurant(self):
        """Test restaurant categorization"""
        from app.ml.categorization_engine import CategoryClassifier

        classifier = CategoryClassifier()
        category, confidence, tax_category = classifier.predict_category(
            description="Lunch",
            merchant="McDonald's"
        )

        assert "Dining" in category
        assert confidence > 0


class TestTaxMapper:
    """Test tax mapper"""

    def test_us_tax_mapping(self):
        """Test US tax mapping"""
        from app.ml.categorization_engine import TaxMapper

        mapper = TaxMapper()
        info = mapper.get_tax_info("Business Expenses", region="US")

        assert info is not None
        assert info["deductible"] == True
        assert "form" in info

    def test_ca_tax_mapping(self):
        """Test Canada tax mapping"""
        from app.ml.categorization_engine import TaxMapper

        mapper = TaxMapper()
        info = mapper.get_tax_info("Business Expenses", region="CA")

        assert info is not None
        assert info["deductible"] == True

    def test_invalid_region(self):
        """Test invalid region"""
        from app.ml.categorization_engine import TaxMapper

        mapper = TaxMapper()
        info = mapper.get_tax_info("Business Expenses", region="INVALID")

        assert info is None
