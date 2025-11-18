"""LLM provider abstraction for multiple backends."""

import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with various LLM providers."""

    def __init__(self, provider: Optional[str] = None):
        """Initialize LLM service.

        Args:
            provider: Override the default provider from settings
        """
        self.provider = provider or settings.llm_provider
        self.client = self._get_client()

    def _get_client(self) -> OpenAI:
        """Get the appropriate OpenAI client based on provider.

        Returns:
            Configured OpenAI client

        Raises:
            ValueError: If provider is unknown
        """
        if self.provider == "lmstudio":
            logger.info(f"Using LM Studio at {settings.lmstudio_base_url}")
            return OpenAI(
                base_url=settings.lmstudio_base_url,
                api_key="lm-studio",  # LM Studio doesn't require real API key
                timeout=settings.request_timeout,
            )
        elif self.provider == "ollama":
            logger.info(f"Using Ollama at {settings.ollama_base_url}/v1")
            return OpenAI(
                base_url=f"{settings.ollama_base_url}/v1",
                api_key="ollama",  # Ollama doesn't require real API key
                timeout=settings.request_timeout,
            )
        elif self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required when using OpenAI provider")
            logger.info("Using OpenAI API")
            return OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.request_timeout,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _get_model_name(self, model_override: Optional[str] = None) -> str:
        """Get the model name based on provider and override.

        Args:
            model_override: Optional model name override

        Returns:
            Model name to use
        """
        if model_override:
            return model_override

        if self.provider == "lmstudio":
            return settings.lmstudio_model
        elif self.provider == "ollama":
            return settings.ollama_model
        elif self.provider == "openai":
            return settings.openai_model
        else:
            return "unknown-model"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional parameters to pass to the API

        Returns:
            Chat completion response dictionary

        Raises:
            Exception: If the API call fails
        """
        model_name = self._get_model_name(model)
        temp = temperature if temperature is not None else settings.default_temperature
        max_tok = max_tokens if max_tokens is not None else settings.default_max_tokens

        logger.info(f"Creating chat completion with {self.provider} using model {model_name}")
        logger.debug(f"Temperature: {temp}, Max tokens: {max_tok}")

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs,
            )

            # Convert to dictionary for easier handling
            result = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            }

            logger.info(f"Chat completion successful. Tokens used: {result['usage']['total_tokens']}")
            return result

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise

    async def extract_receipt(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Extract receipt data using LLM.

        Args:
            messages: Message array for the LLM
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Raw LLM response text

        Raises:
            Exception: If extraction fails
        """
        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response.get("choices"):
            raise ValueError("No choices returned from LLM")

        content = response["choices"][0]["message"]["content"]

        # Clean up common LLM response issues
        content = content.strip()

        # Remove markdown code fences if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]  # Remove ```

        if content.endswith("```"):
            content = content[:-3]  # Remove closing ```

        content = content.strip()

        return content

    async def health_check(self) -> bool:
        """Check if the LLM provider is accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Try a simple completion request
            response = await self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            return bool(response.get("choices"))
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
