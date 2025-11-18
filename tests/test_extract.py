"""
Tests for the extraction endpoint.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_receipts():
    """Load sample receipt fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_receipts.json"
    with open(fixtures_path) as f:
        return json.load(f)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "id": "test-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "merchant": {
                            "name": "Walmart",
                            "store_number": "1234",
                            "address": "123 Main St, Anytown, USA"
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
                            "total": 11.52
                        },
                        "payment_method": "VISA",
                        "currency": "USD"
                    })
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "endpoints" in data
    assert "/api/extract" in data["endpoints"]["extraction"]


@patch('app.services.llm_service.AsyncOpenAI')
def test_extract_endpoint_success(mock_openai, sample_receipts, mock_llm_response):
    """Test successful extraction."""
    # Mock the LLM response
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=type('obj', (object,), {
        'id': mock_llm_response['id'],
        'object': mock_llm_response['object'],
        'created': mock_llm_response['created'],
        'model': mock_llm_response['model'],
        'choices': [type('choice', (object,), {
            'index': 0,
            'message': type('msg', (object,), {
                'role': 'assistant',
                'content': mock_llm_response['choices'][0]['message']['content']
            })(),
            'finish_reason': 'stop'
        })()],
        'usage': type('usage', (object,), {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150
        })()
    })())
    mock_openai.return_value = mock_client

    # Test with walmart sample
    walmart_receipt = sample_receipts["walmart_simple"]
    response = client.post(
        "/api/extract",
        json={"ocr_text": walmart_receipt["ocr_text"]}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["confidence"] > 0.0
    assert data["extraction"] is not None
    assert data["extraction"]["merchant"]["name"] == "Walmart"
    assert len(data["extraction"]["items"]) == 3
    assert data["extraction"]["totals"]["total"] == 11.52


def test_extract_endpoint_missing_ocr_text():
    """Test extraction with missing OCR text."""
    response = client.post(
        "/api/extract",
        json={}
    )

    assert response.status_code == 422  # Validation error


def test_extract_endpoint_empty_ocr_text():
    """Test extraction with empty OCR text."""
    response = client.post(
        "/api/extract",
        json={"ocr_text": ""}
    )

    assert response.status_code == 422  # Validation error (min_length=1)


@pytest.mark.parametrize("receipt_key", [
    "walmart_simple",
    "target_complex",
    "cvs_pharmacy",
    "safeway_messy",
    "staples_office"
])
@patch('app.services.llm_service.AsyncOpenAI')
def test_all_sample_receipts(mock_openai, sample_receipts, mock_llm_response, receipt_key):
    """Test extraction with all sample receipts."""
    # Mock the LLM response
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=type('obj', (object,), {
        'id': mock_llm_response['id'],
        'object': mock_llm_response['object'],
        'created': mock_llm_response['created'],
        'model': mock_llm_response['model'],
        'choices': [type('choice', (object,), {
            'index': 0,
            'message': type('msg', (object,), {
                'role': 'assistant',
                'content': mock_llm_response['choices'][0]['message']['content']
            })(),
            'finish_reason': 'stop'
        })()],
        'usage': type('usage', (object,), {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150
        })()
    })())
    mock_openai.return_value = mock_client

    receipt = sample_receipts[receipt_key]
    response = client.post(
        "/api/extract",
        json={"ocr_text": receipt["ocr_text"]}
    )

    # Should at least not error
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
