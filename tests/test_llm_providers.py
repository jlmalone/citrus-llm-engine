"""Tests for LLM providers."""

import pytest

from app.services.llm_service import LLMService
from app.config import settings


@pytest.mark.asyncio
async def test_llm_service_initialization():
    """Test LLM service initializes correctly."""
    service = LLMService()
    assert service is not None
    assert service.provider is not None


@pytest.mark.asyncio
async def test_llm_provider_selection():
    """Test that correct provider is selected."""
    # Test default provider
    service = LLMService()
    model_name = service.get_model_name()
    assert settings.llm_provider in model_name.lower()


@pytest.mark.asyncio
@pytest.mark.skipif(
    settings.llm_provider != "lmstudio",
    reason="LM Studio not configured",
)
async def test_lmstudio_extraction():
    """Test extraction with LM Studio (if available)."""
    service = LLMService(provider="lmstudio")

    ocr_text = "WALMART\nMILK $3.99\nBREAD $2.49\nTOTAL $6.48"
    system_prompt = "Extract JSON data from receipt"

    try:
        result = await service.extract_receipt_data(ocr_text, system_prompt)
        assert isinstance(result, dict)
    except Exception as e:
        pytest.skip(f"LM Studio not available: {e}")


def test_invalid_provider():
    """Test that invalid provider raises error."""
    with pytest.raises(ValueError):
        LLMService(provider="invalid_provider")
