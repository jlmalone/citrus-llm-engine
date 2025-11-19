"""Main FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import extract, chat, ocr, categorize, tax
from app.models.response import HealthResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown events."""
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"OCR Engine: {settings.ocr_engine}")
    logger.info(f"Tax Assistant: {'enabled' if settings.enable_tax_assistant else 'disabled'}")

    # Initialize services here if needed
    # e.g., load ML models, establish connections

    yield

    # Cleanup on shutdown
    logger.info("Shutting down service")


# Create FastAPI application
app = FastAPI(
    title=settings.service_name,
    version=settings.service_version,
    description="ML/AI receipt extraction service with advanced OCR, LLM extraction, categorization, and tax assistant",
    lifespan=lifespan,
)

# Configure CORS
origins = settings.cors_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.environment == "development" else None,
        },
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.service_version,
        timestamp=datetime.utcnow(),
        services={
            "llm": True,  # TODO: Add actual health checks
            "ocr": True,
            "categorization": True,
            "tax_assistant": settings.enable_tax_assistant,
        },
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "extract": "/api/extract",
            "ocr": "/api/ocr",
            "chat": "/v1/chat/completions",
            "categorize": "/api/categorize",
            "tax": "/api/tax/analyze",
        },
    }


# Include routers
app.include_router(extract.router, prefix="/api", tags=["extraction"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
app.include_router(categorize.router, prefix="/api", tags=["categorization"])
app.include_router(tax.router, prefix="/api/tax", tags=["tax"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
