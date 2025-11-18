"""
Configuration management for citrus-llm-engine.
Uses pydantic-settings for environment-based configuration.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application Settings
    app_name: str = "citrus-llm-engine"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # LLM Provider Configuration
    llm_provider: Literal["lmstudio", "ollama", "openai"] = "lmstudio"
    llm_model: str = "local-model"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000

    # LM Studio Settings
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "not-needed"

    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"

    # OpenAI Settings (fallback)
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    openai_base_url: str = "https://api.openai.com/v1"

    # Receipt Extraction Settings
    default_currency: str = "USD"
    confidence_threshold: float = 0.7

    @property
    def base_url(self) -> str:
        """Get the base URL for the current LLM provider."""
        if self.llm_provider == "lmstudio":
            return self.lmstudio_base_url
        elif self.llm_provider == "ollama":
            return self.ollama_base_url
        else:
            return self.openai_base_url

    @property
    def api_key(self) -> str:
        """Get the API key for the current LLM provider."""
        if self.llm_provider == "lmstudio":
            return self.lmstudio_api_key
        elif self.llm_provider == "ollama":
            return "not-needed"
        else:
            return self.openai_api_key

    @property
    def model_name(self) -> str:
        """Get the model name for the current LLM provider."""
        if self.llm_provider == "lmstudio":
            return self.llm_model
        elif self.llm_provider == "ollama":
            return self.ollama_model
        else:
            return self.openai_model


# Global settings instance
settings = Settings()
