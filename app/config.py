"""Configuration management for Citrus LLM Engine."""

from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service Configuration
    service_name: str = Field(default="citrus-llm-engine", alias="SERVICE_NAME")
    service_version: str = Field(default="1.0.0", alias="SERVICE_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # LLM Provider Configuration
    llm_provider: Literal["lmstudio", "ollama", "openai", "anthropic"] = Field(
        default="lmstudio", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="llama-3-8b-instruct", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")

    # LM Studio
    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL"
    )
    lmstudio_api_key: str = Field(default="lm-studio", alias="LMSTUDIO_API_KEY")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")

    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default="claude-3-sonnet-20240229", alias="ANTHROPIC_MODEL"
    )

    # OCR Configuration
    ocr_engine: Literal["tesseract", "easyocr", "paddleocr", "ensemble"] = Field(
        default="ensemble", alias="OCR_ENGINE"
    )
    ocr_languages: str = Field(default="eng,fra,spa,deu", alias="OCR_LANGUAGES")
    ocr_dpi: int = Field(default=300, alias="OCR_DPI")
    ocr_preprocessing: bool = Field(default=True, alias="OCR_PREPROCESSING")
    ocr_confidence_threshold: float = Field(
        default=0.80, alias="OCR_CONFIDENCE_THRESHOLD"
    )

    # Tesseract
    tesseract_cmd: str = Field(default="/usr/bin/tesseract", alias="TESSERACT_CMD")
    tesseract_lang: str = Field(default="eng", alias="TESSERACT_LANG")

    # EasyOCR
    easyocr_gpu: bool = Field(default=False, alias="EASYOCR_GPU")
    easyocr_languages: str = Field(default="en,fr,es,de", alias="EASYOCR_LANGUAGES")

    # PaddleOCR
    paddleocr_use_gpu: bool = Field(default=False, alias="PADDLEOCR_USE_GPU")
    paddleocr_lang: str = Field(default="en", alias="PADDLEOCR_LANG")
    paddleocr_use_angle_cls: bool = Field(
        default=True, alias="PADDLEOCR_USE_ANGLE_CLS"
    )

    # ML Categorization
    categorization_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="CATEGORIZATION_MODEL",
    )
    categorization_confidence_threshold: float = Field(
        default=0.75, alias="CATEGORIZATION_CONFIDENCE_THRESHOLD"
    )
    enable_auto_categorization: bool = Field(
        default=True, alias="ENABLE_AUTO_CATEGORIZATION"
    )

    # Tax Configuration
    enable_tax_assistant: bool = Field(default=True, alias="ENABLE_TAX_ASSISTANT")
    tax_jurisdictions: str = Field(default="US,CA,GB", alias="TAX_JURISDICTIONS")
    default_tax_jurisdiction: str = Field(default="US", alias="DEFAULT_TAX_JURISDICTION")

    # IRS (US)
    irs_enable_schedule_c: bool = Field(default=True, alias="IRS_ENABLE_SCHEDULE_C")
    irs_enable_schedule_e: bool = Field(default=True, alias="IRS_ENABLE_SCHEDULE_E")
    irs_standard_mileage_rate: float = Field(
        default=0.655, alias="IRS_STANDARD_MILEAGE_RATE"
    )

    # CRA (Canada)
    cra_enable_t2125: bool = Field(default=True, alias="CRA_ENABLE_T2125")
    cra_gst_hst_rate: float = Field(default=0.05, alias="CRA_GST_HST_RATE")

    # HMRC (UK)
    hmrc_enable_sa103: bool = Field(default=True, alias="HMRC_ENABLE_SA103")
    hmrc_vat_standard_rate: float = Field(default=0.20, alias="HMRC_VAT_STANDARD_RATE")

    # Performance
    worker_processes: int = Field(default=4, alias="WORKER_PROCESSES")
    request_timeout: int = Field(default=30, alias="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(
        default=100, alias="MAX_CONCURRENT_REQUESTS"
    )

    # Cache
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")

    # Security
    api_key_required: bool = Field(default=False, alias="API_KEY_REQUIRED")
    api_key: str = Field(default="", alias="API_KEY")
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080", alias="CORS_ORIGINS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
