"""Tests for ML categorization engine."""

import pytest

from app.ml.categorization_engine import CategorizationEngine, CategoryDatabase


def test_category_database():
    """Test category database."""
    keywords = CategoryDatabase.get_all_keywords()
    assert len(keywords) > 0
    assert "coffee" in keywords
    assert keywords["coffee"] == "Food & Beverage"


def test_categorization_engine_init():
    """Test categorization engine initialization."""
    engine = CategorizationEngine()
    assert engine is not None


def test_categorize_item():
    """Test item categorization."""
    engine = CategorizationEngine()

    # Test various items
    test_cases = [
        ("Coffee Latte", "Food & Beverage"),
        ("Office Paper", "Office"),
        ("Gasoline", "Transportation"),
        ("Laptop Computer", "Electronics"),
        ("Legal Services", "Professional Services"),
    ]

    for description, expected_category in test_cases:
        result = engine.categorize(description)
        assert expected_category in result["category"]
        assert result["confidence"] > 0.5
        assert "tax_deductible" in result


def test_categorize_with_merchant():
    """Test categorization with merchant context."""
    engine = CategorizationEngine()

    result = engine.categorize("Latte", merchant="Starbucks")
    assert "Food & Beverage" in result["category"]
    assert result["tax_deductible"] is True  # Business meals are deductible


def test_categorize_batch():
    """Test batch categorization."""
    engine = CategorizationEngine()

    items = ["Coffee", "Paper", "Gas", "Laptop"]
    results = engine.categorize_batch(items)

    assert len(results) == 4
    for result in results:
        assert "category" in result
        assert "confidence" in result


def test_tax_category_mapping():
    """Test tax category mapping."""
    engine = CategorizationEngine()

    # Test US mapping
    irs_category = engine.get_tax_category("Office > Supplies", "US")
    assert irs_category is not None
    assert "Office" in irs_category

    # Test CA mapping
    cra_category = engine.get_tax_category("Transportation > Fuel", "CA")
    assert cra_category is not None

    # Test GB mapping
    hmrc_category = engine.get_tax_category("Professional Services", "GB")
    assert hmrc_category is not None
