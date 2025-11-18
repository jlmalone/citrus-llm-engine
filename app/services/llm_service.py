"""
LLM service for interacting with different LLM providers.
Supports LM Studio, Ollama, and OpenAI with a unified interface.
"""

import json
import logging
from typing import Optional, Any
from openai import OpenAI, AsyncOpenAI
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with LLM providers."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM service.

        Args:
            provider: LLM provider to use (lmstudio, ollama, openai). Uses settings.llm_provider if not specified.
            model: Model name to use. Uses provider-specific default if not specified.
        """
        self.provider = provider or settings.llm_provider
        self.model = model or settings.model_name

        # Initialize the appropriate client based on provider
        if self.provider == "ollama":
            # Ollama uses a different endpoint structure
            self.client = None
            self.base_url = settings.ollama_base_url
        else:
            # LM Studio and OpenAI use OpenAI-compatible API
            self.client = OpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key
            )
            self.async_client = AsyncOpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key
            )

        logger.info(f"Initialized LLM service: provider={self.provider}, model={self.model}")

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        Generate a chat completion using the configured LLM provider.

        Args:
            messages: List of chat messages in OpenAI format
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            The completion response

        Raises:
            Exception: If the LLM request fails
        """
        temperature = temperature or settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens

        try:
            if self.provider == "ollama":
                return await self._ollama_completion(messages, temperature, max_tokens)
            else:
                return await self._openai_compatible_completion(messages, temperature, max_tokens)
        except Exception as e:
            logger.error(f"LLM completion failed: {str(e)}")
            raise

    async def _openai_compatible_completion(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int
    ) -> dict:
        """
        Generate completion using OpenAI-compatible API (LM Studio, OpenAI).

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Completion response
        """
        logger.debug(f"Calling {self.provider} with model={self.model}")

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    async def _ollama_completion(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int
    ) -> dict:
        """
        Generate completion using Ollama API.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Completion response in OpenAI format
        """
        logger.debug(f"Calling Ollama with model={self.model}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        # Convert Ollama response to OpenAI format
        import time
        created_at = data.get("created_at", "")
        created_timestamp = int(time.time()) if created_at else 0

        return {
            "id": "ollama-" + str(hash(created_at)),
            "object": "chat.completion",
            "created": created_timestamp,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.get("message", {}).get("content", "")
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
        }

    def extract_text_from_completion(self, completion: dict) -> str:
        """
        Extract the text content from a completion response.

        Args:
            completion: The completion response

        Returns:
            The extracted text content
        """
        try:
            return completion["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract text from completion: {str(e)}")
            raise ValueError("Invalid completion response format")

    def parse_json_response(self, text: str) -> dict:
        """
        Parse JSON from LLM response, handling markdown code fences.

        Args:
            text: The text response from LLM

        Returns:
            Parsed JSON data

        Raises:
            ValueError: If JSON parsing fails
        """
        # Remove markdown code fences if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}\nResponse text: {text}")
            raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
