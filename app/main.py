"""
Citrus LLM Engine - FastAPI Application
"""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from app.config import settings
from app.routers import extract, chat, ocr, categorize, tax_assistant
from app.models.response import HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Startup time for uptime calculation
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"OCR Engines: {settings.OCR_ENGINES}")
    yield
    logger.info("Shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced ML/AI service for receipt extraction, OCR, categorization, and tax assistance",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            error_code="INTERNAL_ERROR",
            timestamp=datetime.utcnow(),
        ).model_dump(),
    )


# Include routers
app.include_router(extract.router)
app.include_router(chat.router)
app.include_router(ocr.router)
app.include_router(categorize.router)
app.include_router(tax_assistant.router)


@app.get("/", tags=["health"])
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "healthy",
        "endpoints": {
            "extraction": "/api/extract",
            "chat": "/v1/chat/completions",
            "ocr": "/api/ocr",
            "categorization": "/api/categorize",
            "tax_assistant": "/api/tax-assistant",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    # Check service availability
    services = {
        "llm": True,  # Would check LLM provider
        "ocr": True,  # Would check OCR engines
        "ml": True,  # Would check ML models
    }

    uptime = time.time() - startup_time

    return HealthResponse(
        status="healthy" if all(services.values()) else "degraded",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        services=services,
        uptime_seconds=uptime,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
