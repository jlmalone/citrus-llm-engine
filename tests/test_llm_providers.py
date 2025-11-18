"""
Tests for LLM provider service.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return type('obj', (object,), {
        'id': 'chatcmpl-123',
        'object': 'chat.completion',
        'created': 1234567890,
        'model': 'gpt-4',
        'choices': [type('choice', (object,), {
            'index': 0,
            'message': type('msg', (object,), {
                'role': 'assistant',
                'content': 'Hello! How can I help you?'
            })(),
            'finish_reason': 'stop'
        })()],
        'usage': type('usage', (object,), {
            'prompt_tokens': 10,
            'completion_tokens': 8,
            'total_tokens': 18
        })()
    })()


def test_llm_service_initialization():
    """Test LLM service initialization with different providers."""
    # Test default provider
    service = LLMService()
    assert service.provider in ["lmstudio", "ollama", "openai"]

    # Test explicit provider
    service = LLMService(provider="lmstudio", model="test-model")
    assert service.provider == "lmstudio"
    assert service.model == "test-model"


@patch('app.services.llm_service.AsyncOpenAI')
@pytest.mark.asyncio
async def test_openai_compatible_completion(mock_openai, sample_messages, mock_openai_response):
    """Test completion with OpenAI-compatible provider (LM Studio)."""
    # Mock the client
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
    mock_openai.return_value = mock_client

    # Create service
    service = LLMService(provider="lmstudio")

    # Test completion
    result = await service.chat_completion(sample_messages)

    assert result["id"] == "chatcmpl-123"
    assert result["model"] == "gpt-4"
    assert len(result["choices"]) == 1
    assert result["choices"][0]["message"]["content"] == "Hello! How can I help you?"
    assert result["usage"]["total_tokens"] == 18


@patch('httpx.AsyncClient')
@pytest.mark.asyncio
async def test_ollama_completion(mock_httpx, sample_messages):
    """Test completion with Ollama provider."""
    # Mock Ollama response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "llama2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": "Hello from Ollama!"
        },
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 5
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()

    mock_httpx.return_value = mock_client

    # Create service
    service = LLMService(provider="ollama", model="llama2")

    # Test completion
    result = await service.chat_completion(sample_messages)

    assert result["model"] == "llama2"
    assert len(result["choices"]) == 1
    assert result["choices"][0]["message"]["content"] == "Hello from Ollama!"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5


def test_extract_text_from_completion():
    """Test extracting text from completion response."""
    service = LLMService()

    completion = {
        "choices": [
            {
                "message": {
                    "content": "Test content"
                }
            }
        ]
    }

    text = service.extract_text_from_completion(completion)
    assert text == "Test content"


def test_extract_text_from_completion_error():
    """Test extracting text from invalid completion."""
    service = LLMService()

    with pytest.raises(ValueError):
        service.extract_text_from_completion({"invalid": "structure"})


def test_parse_json_response():
    """Test parsing JSON from LLM response."""
    service = LLMService()

    # Test plain JSON
    result = service.parse_json_response('{"test": "value"}')
    assert result == {"test": "value"}

    # Test JSON with markdown fences
    result = service.parse_json_response('```json\n{"test": "value"}\n```')
    assert result == {"test": "value"}

    # Test JSON with generic fences
    result = service.parse_json_response('```\n{"test": "value"}\n```')
    assert result == {"test": "value"}


def test_parse_json_response_error():
    """Test parsing invalid JSON."""
    service = LLMService()

    with pytest.raises(ValueError):
        service.parse_json_response('not valid json')


def test_prompt_builder():
    """Test prompt builder functionality."""
    # Test system prompt
    system_prompt = PromptBuilder.get_extraction_system_prompt()
    assert "receipt data extraction" in system_prompt.lower()
    assert "JSON" in system_prompt

    # Test user prompt
    user_prompt = PromptBuilder.build_extraction_user_prompt("WALMART\nMILK $3.99")
    assert "WALMART" in user_prompt
    assert "MILK $3.99" in user_prompt

    # Test chat messages
    messages = PromptBuilder.build_chat_messages("WALMART\nMILK $3.99")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
