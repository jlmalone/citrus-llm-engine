"""Tests for extraction endpoint."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_receipts():
    """Load sample receipts from fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_receipts.json"
    with open(fixtures_path) as f:
        return json.load(f)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_extract_with_ocr_text(sample_receipts):
    """Test extraction with pre-extracted OCR text."""
    walmart = sample_receipts["walmart_receipt"]

    response = client.post(
        "/api/extract",
        json={
            "ocr_text": walmart["ocr_text"],
            "enable_categorization": True,
            "enable_tax_analysis": True,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["confidence"] > 0.5
    assert "extraction" in data
    assert "merchant" in data["extraction"]


def test_extract_without_input():
    """Test extraction without any input."""
    response = client.post(
        "/api/extract",
        json={},
    )

    assert response.status_code == 400


def test_categorization(sample_receipts):
    """Test categorization endpoint."""
    response = client.post(
        "/api/categorize",
        json={
            "items": ["Coffee Latte", "Office Paper", "Gasoline", "Laptop Computer"],
            "merchant_name": "General Store",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["count"] == 4
    assert len(data["results"]) == 4

    # Check that items were categorized
    for result in data["results"]:
        assert "category" in result
        assert "confidence" in result
        assert "tax_deductible" in result


def test_tax_analysis():
    """Test tax analysis endpoint."""
    response = client.post(
        "/api/tax/analyze",
        json={
            "merchant_name": "Starbucks",
            "items": [
                {"description": "Coffee", "price": 4.95, "category": "Food & Beverage"},
                {"description": "Muffin", "price": 3.45, "category": "Food & Beverage"},
            ],
            "total_amount": 9.07,
            "jurisdiction": "US",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["jurisdiction"] == "US"
    assert data["total_amount"] == 9.07
    assert "deductible_amount" in data
    assert "deductions" in data
    assert len(data["deductions"]) == 2


def test_tax_analysis_multiple_jurisdictions():
    """Test tax analysis for different jurisdictions."""
    items = [
        {"description": "Office supplies", "price": 50.00, "category": "Office"},
    ]

    for jurisdiction in ["US", "CA", "GB"]:
        response = client.post(
            "/api/tax/analyze",
            json={
                "merchant_name": "Office Store",
                "items": items,
                "total_amount": 50.00,
                "jurisdiction": jurisdiction,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jurisdiction"] == jurisdiction
