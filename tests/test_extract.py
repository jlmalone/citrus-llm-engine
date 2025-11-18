"""Tests for extraction endpoint."""

import json
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.llm_service import LLMService

client = TestClient(app)


@pytest.fixture
def sample_receipt_ocr():
    """Sample receipt OCR text."""
    return """WAL*MART #1234
123 Main St
12/25/2024 3:45 PM

MILK 2% GAL    $3.99
BREAD WHEAT    $2.49
EGGS DOZEN     $4.29

SUBTOTAL      $10.77
TAX            $0.75
TOTAL         $11.52"""


@pytest.fixture
def sample_llm_response():
    """Sample LLM extraction response."""
    return json.dumps({
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
            "tip": None,
            "discount": None
        },
        "payment_method": None,
        "currency": "USD"
    })


def test_root_endpoint():
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test health check endpoint."""
    with patch.object(LLMService, "health_check", new_callable=AsyncMock, return_value=True):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


@pytest.mark.asyncio
async def test_extract_success(sample_receipt_ocr, sample_llm_response):
    """Test successful receipt extraction."""
    with patch.object(LLMService, "extract_receipt", new_callable=AsyncMock, return_value=sample_llm_response):
        response = client.post(
            "/api/extract",
            json={"ocr_text": sample_receipt_ocr}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["extraction"] is not None
        assert data["extraction"]["merchant"]["name"] == "Walmart"
        assert len(data["extraction"]["items"]) == 3
        assert data["extraction"]["totals"]["total"] == 11.52


@pytest.mark.asyncio
async def test_extract_invalid_json():
    """Test extraction with invalid JSON response."""
    with patch.object(LLMService, "extract_receipt", new_callable=AsyncMock, return_value="Not valid JSON"):
        response = client.post(
            "/api/extract",
            json={"ocr_text": "Some receipt text"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is False
        assert "Failed to parse" in data["error"]
        assert data["raw_response"] == "Not valid JSON"


@pytest.mark.asyncio
async def test_extract_with_model_override(sample_receipt_ocr, sample_llm_response):
    """Test extraction with model override."""
    with patch.object(LLMService, "extract_receipt", new_callable=AsyncMock, return_value=sample_llm_response):
        response = client.post(
            "/api/extract",
            json={
                "ocr_text": sample_receipt_ocr,
                "model": "custom-model"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


@pytest.mark.asyncio
async def test_extract_with_temperature(sample_receipt_ocr, sample_llm_response):
    """Test extraction with temperature override."""
    with patch.object(LLMService, "extract_receipt", new_callable=AsyncMock, return_value=sample_llm_response):
        response = client.post(
            "/api/extract",
            json={
                "ocr_text": sample_receipt_ocr,
                "temperature": 0.5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


def test_extract_missing_ocr_text():
    """Test extraction without OCR text."""
    response = client.post(
        "/api/extract",
        json={}
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_chat_completions_endpoint():
    """Test OpenAI-compatible chat completions endpoint."""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

    with patch.object(LLMService, "chat_completion", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "chatcmpl-123"
        assert data["choices"][0]["message"]["content"] == "Test response"


def test_chat_completions_streaming_not_supported():
    """Test that streaming is not supported."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
    )

    assert response.status_code == 400
    assert "not yet supported" in response.json()["detail"]
