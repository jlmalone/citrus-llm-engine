# 🍋 Citrus LLM Engine

**Advanced ML/AI microservice for receipt extraction, OCR, categorization, and tax assistance**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Features

### 🧠 LLM-Powered Receipt Extraction
- Extract structured JSON from raw OCR text
- Support for multiple LLM providers:
  - **LM Studio** (local, privacy-first)
  - **Ollama** (local, open-source)
  - **OpenAI** (cloud fallback)
- Intelligent field extraction and normalization
- Merchant name cleaning and standardization

### 👁️ Advanced OCR (99%+ Accuracy)
- **Triple-engine ensemble** for maximum accuracy:
  - **Tesseract OCR** - Fast and reliable
  - **EasyOCR** - Deep learning-based
  - **PaddleOCR** - High accuracy for complex layouts
- Intelligent image preprocessing
- Multiple ensemble strategies (voting, best, consensus)
- Multi-language support

### 🏷️ ML Categorization Engine
- Automatic expense categorization
- Hierarchical category structure
- Tax category mapping (IRS/CRA/HMRC)
- LLM fallback for edge cases
- High-confidence predictions

### 💼 Tax Assistant
- Multi-region support:
  - **🇺🇸 IRS** (United States)
  - **🇨🇦 CRA** (Canada)
  - **🇬🇧 HMRC** (United Kingdom)
- Deduction identification and analysis
- Documentation requirements
- Compliance checking
- Percentage calculations (partial deductions)

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Architecture](#-architecture)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional but recommended)
- LM Studio or Ollama (for local LLM)

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/citrus-llm-engine.git
cd citrus-llm-engine

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Run with Docker (Recommended)

```bash
# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Service is now running at http://localhost:8000
```

### 3. Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --reload

# Service is now running at http://localhost:8000
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Extract receipt data
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_text": "WALMART #1234\nMILK $3.99\nBREAD $2.49\nTOTAL $6.48"
  }'
```

## 🔧 Installation

### Option 1: Docker (Recommended)

```bash
docker-compose up -d
```

### Option 2: Poetry

```bash
poetry install
poetry run uvicorn app.main:app --reload
```

### Option 3: pip

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### OCR Engine Setup

**Tesseract:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

**EasyOCR and PaddleOCR** are installed via pip requirements.

## 📚 API Documentation

### Interactive Docs

Once the service is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/extract` | POST | Extract structured data from receipt text |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions |
| `/api/ocr` | POST | Extract text from receipt images |
| `/api/categorize` | POST | Categorize expenses with ML/AI |
| `/api/tax-assistant` | POST | Analyze tax deductibility |
| `/health` | GET | Health check |

## 🎯 API Examples

### 1. Receipt Extraction

**Request:**
```json
POST /api/extract

{
  "ocr_text": "WAL*MART #1234\n123 Main St\n12/25/2024 3:45 PM\n\nMILK 2% GAL    $3.99\nBREAD WHEAT    $2.49\nEGGS DOZEN     $4.29\n\nSUBTOTAL      $10.77\nTAX            $0.75\nTOTAL         $11.52",
  "model": "llama-3-8b-instruct"
}
```

**Response:**
```json
{
  "success": true,
  "confidence": 0.95,
  "extraction": {
    "merchant": {
      "name": "Walmart",
      "store_number": "1234",
      "address": "123 Main St"
    },
    "timestamp": "2024-12-25T15:45:00Z",
    "items": [
      {
        "description": "Milk 2% Gallon",
        "price": 3.99,
        "quantity": 1,
        "category": "Groceries > Dairy"
      }
    ],
    "totals": {
      "subtotal": 10.77,
      "tax": 0.75,
      "total": 11.52
    },
    "currency": "USD"
  },
  "provider": "lmstudio",
  "processing_time_ms": 1250
}
```

### 2. Advanced OCR

**Request:**
```json
POST /api/ocr

{
  "image_data": "data:image/png;base64,iVBORw0KGgoAAAANS...",
  "engines": ["tesseract", "easyocr", "paddleocr"],
  "preprocessing": true,
  "ensemble_mode": "voting"
}
```

### 3. Expense Categorization

**Request:**
```json
POST /api/categorize

{
  "description": "Office supplies from Staples",
  "amount": 45.99,
  "merchant": "Staples",
  "tax_mapping": true
}
```

### 4. Tax Assistant

**Request:**
```json
POST /api/tax-assistant

{
  "description": "Business lunch with client",
  "amount": 85.00,
  "category": "Meals",
  "region": "US",
  "tax_year": 2024,
  "business_use": true
}
```

**Response:**
```json
{
  "success": true,
  "region": "US",
  "tax_year": 2024,
  "deduction": {
    "deduction_type": "partially_deductible",
    "deductible_amount": 42.50,
    "deductible_percentage": 50.0,
    "category": "Meals and Entertainment",
    "irs_category": "Meals and Entertainment"
  },
  "confidence": 0.85
}
```

## ⚙️ Configuration

See [.env.example](.env.example) for all configuration options.

**Key settings:**

```bash
# LLM Provider
LLM_PROVIDER=lmstudio          # lmstudio, ollama, openai
LLM_MODEL=llama-3-8b-instruct

# OCR Engines
OCR_ENGINES=tesseract,easyocr,paddleocr
OCR_CONFIDENCE_THRESHOLD=0.80

# Tax Regions
TAX_REGIONS=US,CA,GB
TAX_YEAR=2024
```

## 🏗️ Architecture

```
citrus-llm-engine/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── routers/                # API endpoints
│   │   ├── extract.py          # Receipt extraction
│   │   ├── chat.py             # Chat completions
│   │   ├── ocr.py              # OCR processing
│   │   ├── categorize.py       # Categorization
│   │   └── tax_assistant.py    # Tax analysis
│   ├── services/               # Business logic
│   │   ├── llm_service.py      # LLM provider abstraction
│   │   ├── ocr_service.py      # OCR engine ensemble
│   │   └── prompt_builder.py   # Prompt engineering
│   ├── ml/                     # ML models
│   │   ├── categorization_engine.py
│   │   └── tax_assistant.py
│   └── models/                 # Pydantic models
├── tests/                      # Unit tests
├── Dockerfile                  # Container definition
└── docker-compose.yml          # Docker Compose config
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_extract.py
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production Deployment

```bash
export DEBUG=false
export LOG_LEVEL=INFO
export WORKERS=4

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 📊 Performance

- **Extraction**: ~1-2s per receipt
- **OCR**: ~3-5s per image (ensemble mode)
- **Categorization**: <100ms
- **Tax Analysis**: ~200ms
- Supports horizontal scaling

## 🛠️ Development

```bash
# Format code
black app/ tests/

# Run tests
pytest
```

## 📝 License

MIT License - see LICENSE file for details

---

**Built with ❤️ for the Citrus receipt processing ecosystem**

Part of the [Citrus Platform](https://github.com/jlmalone/citrus-web-modern)
