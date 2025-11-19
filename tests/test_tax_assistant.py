"""Tests for tax assistant."""

import pytest

from app.ml.tax_assistant import TaxAssistant, IRSTaxRules, CRATaxRules, HMRCTaxRules


def test_tax_assistant_init():
    """Test tax assistant initialization."""
    assistant = TaxAssistant()
    assert assistant is not None
    assert len(assistant.rules) == 3


def test_irs_rules():
    """Test IRS tax rules."""
    rules = IRSTaxRules()

    # Test meal deduction (50%)
    item = {
        "description": "Coffee",
        "price": 10.00,
        "category": "Food & Beverage > Coffee",
    }
    result = rules.evaluate(item, {})
    assert result["deductible"] is True
    assert result["percentage"] == 50.0

    # Test office supplies (100%)
    item = {
        "description": "Paper",
        "price": 25.00,
        "category": "Office > Supplies",
    }
    result = rules.evaluate(item, {})
    assert result["deductible"] is True
    assert result["percentage"] == 100.0


def test_cra_rules():
    """Test CRA tax rules."""
    rules = CRATaxRules()

    # Test meal deduction (50%)
    item = {
        "description": "Lunch",
        "price": 15.00,
        "category": "Food & Beverage > Restaurant",
    }
    result = rules.evaluate(item, {})
    assert result["deductible"] is True
    assert result["percentage"] == 50.0


def test_hmrc_rules():
    """Test HMRC tax rules."""
    rules = HMRCTaxRules()

    # Test business meal
    item = {
        "description": "Business lunch",
        "price": 20.00,
        "category": "Food & Beverage > Restaurant",
    }
    result = rules.evaluate(item, {})
    assert result["deductible"] is True


def test_analyze_receipt_us():
    """Test receipt analysis for US."""
    assistant = TaxAssistant()

    items = [
        {"description": "Coffee", "price": 5.00, "category": "Food & Beverage"},
        {"description": "Office paper", "price": 25.00, "category": "Office"},
        {"description": "Gasoline", "price": 50.00, "category": "Transportation"},
    ]

    analysis = assistant.analyze_receipt(
        merchant="Test Store",
        items=items,
        total_amount=80.00,
        jurisdiction="US",
    )

    assert analysis["jurisdiction"] == "US"
    assert analysis["total_amount"] == 80.00
    assert len(analysis["deductions"]) == 3
    assert analysis["deductible_amount"] > 0
    assert 0 <= analysis["deductible_percentage"] <= 100


def test_analyze_receipt_ca():
    """Test receipt analysis for Canada."""
    assistant = TaxAssistant()

    items = [
        {"description": "Supplies", "price": 100.00, "category": "Office"},
    ]

    analysis = assistant.analyze_receipt(
        merchant="Canadian Store",
        items=items,
        total_amount=100.00,
        jurisdiction="CA",
    )

    assert analysis["jurisdiction"] == "CA"
    assert analysis["deductible_amount"] == 100.00


def test_analyze_receipt_gb():
    """Test receipt analysis for UK."""
    assistant = TaxAssistant()

    items = [
        {"description": "Equipment", "price": 200.00, "category": "Office"},
    ]

    analysis = assistant.analyze_receipt(
        merchant="UK Store",
        items=items,
        total_amount=200.00,
        jurisdiction="GB",
    )

    assert analysis["jurisdiction"] == "GB"
    assert "recommendations" in analysis
    assert "warnings" in analysis


def test_invalid_jurisdiction():
    """Test invalid jurisdiction raises error."""
    assistant = TaxAssistant()

    with pytest.raises(ValueError):
        assistant.analyze_receipt(
            merchant="Store",
            items=[],
            total_amount=0,
            jurisdiction="XX",
        )
