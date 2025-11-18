"""Tests for LLM provider abstraction."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from app.services.llm_service import LLMService
from app.config import settings


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response."""
    mock_response = Mock()
    mock_response.id = "chatcmpl-123"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = "test-model"

    mock_choice = Mock()
    mock_choice.index = 0
    mock_choice.message = Mock()
    mock_choice.message.role = "assistant"
    mock_choice.message.content = "Test response"
    mock_choice.finish_reason = "stop"

    mock_response.choices = [mock_choice]

    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30

    mock_response.usage = mock_usage

    return mock_response


@pytest.mark.asyncio
async def test_lmstudio_provider():
    """Test LM Studio provider initialization."""
    service = LLMService(provider="lmstudio")
    assert service.provider == "lmstudio"
    assert service.client.base_url == settings.lmstudio_base_url


@pytest.mark.asyncio
async def test_ollama_provider():
    """Test Ollama provider initialization."""
    service = LLMService(provider="ollama")
    assert service.provider == "ollama"
    assert f"{settings.ollama_base_url}/v1" in str(service.client.base_url)


@pytest.mark.asyncio
async def test_openai_provider():
    """Test OpenAI provider initialization."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        from app.config import Settings
        test_settings = Settings(openai_api_key="test-key")

        with patch("app.services.llm_service.settings", test_settings):
            service = LLMService(provider="openai")
            assert service.provider == "openai"


@pytest.mark.asyncio
async def test_invalid_provider():
    """Test invalid provider raises error."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMService(provider="invalid")


@pytest.mark.asyncio
async def test_chat_completion(mock_openai_response):
    """Test chat completion."""
    service = LLMService(provider="lmstudio")

    with patch.object(service.client.chat.completions, "create", return_value=mock_openai_response):
        messages = [{"role": "user", "content": "Hello"}]
        response = await service.chat_completion(messages=messages)

        assert response["id"] == "chatcmpl-123"
        assert response["choices"][0]["message"]["content"] == "Test response"
        assert response["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_extract_receipt(mock_openai_response):
    """Test receipt extraction."""
    service = LLMService(provider="lmstudio")

    mock_openai_response.choices[0].message.content = '{"merchant": {"name": "Test"}}'

    with patch.object(service.client.chat.completions, "create", return_value=mock_openai_response):
        messages = [{"role": "user", "content": "Extract receipt"}]
        response = await service.extract_receipt(messages=messages)

        assert response == '{"merchant": {"name": "Test"}}'


@pytest.mark.asyncio
async def test_extract_receipt_strips_markdown(mock_openai_response):
    """Test receipt extraction strips markdown fences."""
    service = LLMService(provider="lmstudio")

    mock_openai_response.choices[0].message.content = '```json\n{"merchant": {"name": "Test"}}\n```'

    with patch.object(service.client.chat.completions, "create", return_value=mock_openai_response):
        messages = [{"role": "user", "content": "Extract receipt"}]
        response = await service.extract_receipt(messages=messages)

        assert response == '{"merchant": {"name": "Test"}}'
        assert "```" not in response


@pytest.mark.asyncio
async def test_model_name_override():
    """Test model name override."""
    service = LLMService(provider="lmstudio")

    model_name = service._get_model_name("custom-model")
    assert model_name == "custom-model"


@pytest.mark.asyncio
async def test_model_name_from_settings():
    """Test model name from settings."""
    service = LLMService(provider="ollama")

    model_name = service._get_model_name()
    assert model_name == settings.ollama_model


@pytest.mark.asyncio
async def test_health_check_success(mock_openai_response):
    """Test successful health check."""
    service = LLMService(provider="lmstudio")

    with patch.object(service.client.chat.completions, "create", return_value=mock_openai_response):
        healthy = await service.health_check()
        assert healthy is True


@pytest.mark.asyncio
async def test_health_check_failure():
    """Test failed health check."""
    service = LLMService(provider="lmstudio")

    with patch.object(service.client.chat.completions, "create", side_effect=Exception("Connection error")):
        healthy = await service.health_check()
        assert healthy is False
