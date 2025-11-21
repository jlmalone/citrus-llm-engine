"""
Configuration management for Citrus LLM Engine
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Citrus LLM Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # LLM Provider Configuration
    LLM_PROVIDER: str = "lmstudio"  # lmstudio, ollama, openai
    LLM_MODEL: str = "llama-3-8b-instruct"
    LLM_TEMPERATURE: float = 0.1  # Low temperature for consistent extraction
    LLM_MAX_TOKENS: int = 2000
    LLM_TIMEOUT: int = 60

    # LM Studio Configuration
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_API_KEY: str = "lm-studio"

    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # OCR Configuration
    OCR_ENGINES: str = "tesseract,easyocr,paddleocr"  # Comma-separated
    OCR_LANGUAGES: str = "eng"  # Comma-separated language codes
    OCR_CONFIDENCE_THRESHOLD: float = 0.80
    OCR_PREPROCESSING: bool = True
    TESSERACT_PATH: Optional[str] = None

    # ML Categorization
    CATEGORY_MODEL_PATH: str = "models/category_classifier"
    TAX_MAPPING_PATH: str = "app/ml/data/tax_mappings.json"
    CATEGORY_CONFIDENCE_THRESHOLD: float = 0.75
    USE_LLM_CATEGORIZATION: bool = True  # Fallback to LLM if ML fails

    # Tax Assistant
    TAX_REGIONS: str = "US,CA,GB"  # IRS, CRA, HMRC
    TAX_RULES_PATH: str = "app/ml/data/tax_rules"
    TAX_YEAR: int = 2024

    # Database (optional for caching)
    DATABASE_URL: Optional[str] = None
    REDIS_URL: Optional[str] = None

    # Security
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: str = "*"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
