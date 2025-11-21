"""
LLM Service with multi-provider support (LM Studio, Ollama, OpenAI)
"""
import json
import time
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI, AsyncOpenAI
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self):
        self.provider_name = "base"

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Complete a chat conversation"""
        raise NotImplementedError


class LMStudioProvider(LLMProvider):
    """LM Studio provider (OpenAI-compatible)"""

    def __init__(self):
        super().__init__()
        self.provider_name = "lmstudio"
        self.client = OpenAI(
            base_url=settings.LMSTUDIO_BASE_URL,
            api_key=settings.LMSTUDIO_API_KEY,
        )

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Complete using LM Studio"""
        try:
            response = self.client.chat.completions.create(
                model=model or settings.LLM_MODEL,
                messages=messages,
                temperature=temperature or settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            )

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as e:
            logger.error(f"LM Studio error: {e}")
            raise


class OllamaProvider(LLMProvider):
    """Ollama provider"""

    def __init__(self):
        super().__init__()
        self.provider_name = "ollama"
        self.base_url = settings.OLLAMA_BASE_URL

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Complete using Ollama"""
        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model or settings.OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature or settings.LLM_TEMPERATURE,
                            "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

                return {
                    "content": data["message"]["content"],
                    "model": data["model"],
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": data.get("prompt_eval_count", 0)
                        + data.get("eval_count", 0),
                    },
                    "finish_reason": "stop",
                }
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI provider"""

    def __init__(self):
        super().__init__()
        self.provider_name = "openai"
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Complete using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=model or settings.OPENAI_MODEL,
                messages=messages,
                temperature=temperature or settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            )

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise


class LLMService:
    """Main LLM service with provider abstraction"""

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available providers"""
        try:
            self.providers["lmstudio"] = LMStudioProvider()
            logger.info("LM Studio provider initialized")
        except Exception as e:
            logger.warning(f"LM Studio provider not available: {e}")

        try:
            self.providers["ollama"] = OllamaProvider()
            logger.info("Ollama provider initialized")
        except Exception as e:
            logger.warning(f"Ollama provider not available: {e}")

        try:
            if settings.OPENAI_API_KEY:
                self.providers["openai"] = OpenAIProvider()
                logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"OpenAI provider not available: {e}")

        if not self.providers:
            logger.error("No LLM providers available!")

    def get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        """Get provider instance"""
        provider = provider_name or settings.LLM_PROVIDER

        if provider not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(
                f"Provider '{provider}' not available. Available: {available}"
            )

        return self.providers[provider]

    async def complete(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Complete a chat conversation"""
        start_time = time.time()

        provider_instance = self.get_provider(provider)
        result = await provider_instance.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        processing_time = int((time.time() - start_time) * 1000)
        result["provider"] = provider_instance.provider_name
        result["processing_time_ms"] = processing_time

        return result

    async def extract_json(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        result = await self.complete(
            messages=messages,
            provider=provider,
            model=model,
            temperature=0.1,  # Low temperature for consistent JSON
        )

        content = result["content"]

        # Remove markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Parse JSON
        try:
            parsed = json.loads(content)
            result["parsed_json"] = parsed
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nContent: {content}")
            raise ValueError(f"LLM did not return valid JSON: {e}")


# Global service instance
llm_service = LLMService()
