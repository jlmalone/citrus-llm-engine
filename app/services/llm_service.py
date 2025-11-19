"""LLM service with swappable provider support."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate completion from messages."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass


class LMStudioProvider(LLMProvider):
    """LM Studio provider (local)."""

    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
        )
        self.model = settings.llm_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate completion using LM Studio."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LM Studio error: {e}")
            raise

    def get_model_name(self) -> str:
        return f"lmstudio/{self.model}"


class OllamaProvider(LLMProvider):
    """Ollama provider (local)."""

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate completion using Ollama."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"


class OpenAIProvider(LLMProvider):
    """OpenAI provider (cloud)."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate completion using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise

    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (cloud)."""

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        self.api_key = settings.anthropic_api_key
        self.model = settings.anthropic_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate completion using Anthropic."""
        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": system_message,
                        "messages": anthropic_messages,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            raise

    def get_model_name(self) -> str:
        return f"anthropic/{self.model}"


class LLMService:
    """Service for LLM operations with provider abstraction."""

    def __init__(self, provider: Optional[str] = None) -> None:
        """Initialize LLM service with specified provider."""
        provider_name = provider or settings.llm_provider
        self.provider = self._create_provider(provider_name)
        logger.info(f"Initialized LLM service with provider: {provider_name}")

    def _create_provider(self, provider_name: str) -> LLMProvider:
        """Create provider instance."""
        providers = {
            "lmstudio": LMStudioProvider,
            "ollama": OllamaProvider,
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
        }

        provider_class = providers.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")

        return provider_class()

    async def extract_receipt_data(
        self,
        ocr_text: str,
        system_prompt: str,
    ) -> Dict[str, Any]:
        """Extract structured data from receipt OCR text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ocr_text},
        ]

        response = await self.provider.complete(
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        # Clean response (remove markdown fences if present)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Parse JSON
        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response: {response}")
            raise ValueError(f"LLM returned invalid JSON: {e}")

    async def categorize_item(
        self,
        item_description: str,
        merchant: Optional[str] = None,
        prompt: str = "",
    ) -> Dict[str, Any]:
        """Categorize a single item using LLM."""
        messages = [
            {"role": "system", "content": "You are an expert at categorizing receipt items."},
            {"role": "user", "content": prompt},
        ]

        response = await self.provider.complete(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )

        # Clean and parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse categorization response: {response}")
            return {
                "category": "Uncategorized",
                "confidence": 0.5,
                "tax_relevant": False,
                "business_expense_category": None,
            }

    async def analyze_tax_deductions(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """Analyze receipt for tax deductions."""
        messages = [
            {"role": "system", "content": "You are a tax expert analyzing receipts for deductions."},
            {"role": "user", "content": prompt},
        ]

        response = await self.provider.complete(
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )

        # Clean and parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tax analysis response: {response}")
            return {
                "deductible_amount": 0,
                "deductible_percentage": 0,
                "deductions": [],
                "recommendations": [],
                "warnings": ["Failed to parse tax analysis"],
            }

    def get_model_name(self) -> str:
        """Get the current model name."""
        return self.provider.get_model_name()
