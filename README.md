# 🧠 citrus-llm-engine

**LLM-powered receipt extraction service with OpenAI-compatible interface**

Extract structured JSON from receipt OCR text using local LLMs (LM Studio, Ollama) or cloud LLMs (OpenAI). Built for the [Citrus Platform](https://github.com/jlmalone/citrus-web-modern) receipt processing ecosystem.

---

## 🎯 Features

- **Local-First LLM Support**: LM Studio, Ollama for privacy and cost savings
- **OpenAI Fallback**: Cloud-based option for production workloads
- **Swappable Providers**: Switch between LLMs via environment variables
- **Structured Extraction**: OCR text → Clean JSON with merchant, items, totals
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Smart Parsing**: Handles messy OCR, normalizes merchant names, infers categories
- **Docker Ready**: Single command deployment
- **Type Safe**: Full Pydantic validation and Python 3.11+ type hints

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [LLM Providers](#-llm-providers)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Examples](#-examples)
- [Architecture](#-architecture)

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** or **Docker**
2. **LLM Provider** (choose one):
   - [LM Studio](https://lmstudio.ai/) (recommended for local)
   - [Ollama](https://ollama.ai/) (alternative local)
   - OpenAI API key (cloud fallback)

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/jlmalone/citrus-llm-engine.git
cd citrus-llm-engine

# 2. Configure environment
cp .env.example .env
# Edit .env and set LLM_PROVIDER (lmstudio, ollama, or openai)

# 3. Start LM Studio (if using local LLM)
# - Download from https://lmstudio.ai/
# - Load a model (e.g., llama-3-8b-instruct)
# - Start local server on port 1234

# 4. Start the service
docker-compose up

# 5. Test the service
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
# 1. Clone and setup
git clone https://github.com/jlmalone/citrus-llm-engine.git
cd citrus-llm-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env as needed

# 5. Run the service
python -m uvicorn app.main:app --reload

# Service runs on http://localhost:8000
```

---

## 📚 API Documentation

### Base URL

```
http://localhost:8000
```

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `POST /api/extract` - Extract Receipt Data

Convenience endpoint for extracting structured data from receipt OCR text.

**Request:**

```json
{
  "ocr_text": "WAL*MART #1234\n123 Main St\n12/25/2024 3:45 PM\n\nMILK 2% GAL    $3.99\nBREAD WHEAT    $2.49\nEGGS DOZEN     $4.29\n\nSUBTOTAL      $10.77\nTAX            $0.75\nTOTAL         $11.52",
  "image_url": "optional-for-vision-models",
  "model": "optional-model-override"
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
      },
      {
        "description": "Bread Wheat",
        "price": 2.49,
        "quantity": 1,
        "category": "Groceries > Bakery"
      },
      {
        "description": "Eggs Dozen",
        "price": 4.29,
        "quantity": 1,
        "category": "Groceries > Dairy"
      }
    ],
    "totals": {
      "subtotal": 10.77,
      "tax": 0.75,
      "total": 11.52
    },
    "payment_method": null,
    "currency": "USD"
  }
}
```

#### `POST /v1/chat/completions` - OpenAI-Compatible Chat

Standard OpenAI chat completions endpoint for maximum flexibility.

**Request:**

```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Response:** Standard OpenAI chat completion format

#### `GET /health` - Health Check

Returns service health status.

#### `GET /` - Service Info

Returns service information and available endpoints.

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# LLM Provider (lmstudio, ollama, openai)
LLM_PROVIDER=lmstudio

# LM Studio (default - local)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LLM_MODEL=local-model

# Ollama (alternative local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# OpenAI (cloud fallback)
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=gpt-4

# Application Settings
LOG_LEVEL=INFO
PORT=8000
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `lmstudio` | Provider: `lmstudio`, `ollama`, `openai` |
| `LLM_MODEL` | `local-model` | Model name |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature (0-2) |
| `LLM_MAX_TOKENS` | `2000` | Max tokens to generate |
| `DEFAULT_CURRENCY` | `USD` | Default currency code |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence threshold |

---

## 🤖 LLM Providers

### LM Studio (Recommended for Local)

1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Load a model (recommended: `llama-3-8b-instruct`, `mistral-7b-instruct`)
3. Start local server (Settings → Local Server → Start)
4. Set environment:

```bash
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

**Pros**: Fast, private, free, GUI
**Cons**: Requires ~8GB RAM

### Ollama (Alternative Local)

1. Install from [ollama.ai](https://ollama.ai/)
2. Pull a model: `ollama pull llama2`
3. Run: `ollama serve`
4. Set environment:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama2
```

**Pros**: CLI-focused, lightweight
**Cons**: Requires model management

### OpenAI (Cloud Fallback)

1. Get API key from [platform.openai.com](https://platform.openai.com/api-keys)
2. Set environment:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4
```

**Pros**: Best accuracy, no local resources
**Cons**: Costs per request, data leaves local network

---

## 🛠️ Development

### Project Structure

```
citrus-llm-engine/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models/
│   │   ├── request.py       # Request models
│   │   └── response.py      # Response models
│   ├── routers/
│   │   ├── extract.py       # /api/extract endpoint
│   │   └── chat.py          # /v1/chat/completions endpoint
│   └── services/
│       ├── llm_service.py   # LLM provider abstraction
│       └── prompt_builder.py # System prompt engineering
├── tests/
│   ├── test_extract.py      # Extraction tests
│   ├── test_llm_providers.py # Provider tests
│   └── fixtures/
│       └── sample_receipts.json # Test data
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Running Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Run with auto-reload
python -m uvicorn app.main:app --reload --log-level debug

# Or use the main.py entry point
python app/main.py
```

---

## 🧪 Testing

### Run Tests

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_extract.py -v
```

### Test Fixtures

Sample receipts are in `tests/fixtures/sample_receipts.json`:

- Walmart (simple)
- Target (complex)
- CVS Pharmacy (medical)
- Safeway (messy OCR)
- Staples (office supplies)

---

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

### Production Considerations

1. **API Keys**: Use secrets management (not .env files)
2. **CORS**: Configure `allow_origins` in `app/main.py`
3. **Rate Limiting**: Add rate limiting middleware
4. **Monitoring**: Add logging/metrics (Prometheus, DataDog)
5. **Scaling**: Deploy multiple instances behind load balancer
6. **Model Size**: Optimize for target hardware (8B models for 8GB RAM)

---

## 📝 Examples

### cURL Examples

```bash
# Extract receipt data
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_text": "WALMART #1234\nMILK $3.99\nBREAD $2.49\nTOTAL $6.48"
  }'

# Health check
curl http://localhost:8000/health

# Service info
curl http://localhost:8000/
```

### Python Client Example

```python
import requests

# Extract receipt
response = requests.post(
    "http://localhost:8000/api/extract",
    json={
        "ocr_text": "WALMART #1234\nMILK $3.99\nBREAD $2.49\nTOTAL $6.48"
    }
)

data = response.json()
if data["success"]:
    merchant = data["extraction"]["merchant"]["name"]
    total = data["extraction"]["totals"]["total"]
    print(f"{merchant}: ${total}")
```

### Integration with Citrus Backend (Kotlin/Ktor)

```kotlin
// In your Ktor service
val client = HttpClient()

val response = client.post("http://citrus-llm-engine:8000/api/extract") {
    contentType(ContentType.Application.Json)
    setBody(mapOf("ocr_text" to ocrText))
}

val extraction: ExtractionResponse = response.body()
if (extraction.success) {
    // Save to database
    saveReceipt(extraction.extraction)
}
```

---

## 🏗️ Architecture

### System Flow

```
Receipt Image
    ↓
OCR Service (Tesseract/Cloud)
    ↓
OCR Text
    ↓
citrus-llm-engine (/api/extract)
    ↓
Prompt Builder (system + user prompt)
    ↓
LLM Service (LM Studio/Ollama/OpenAI)
    ↓
Structured JSON Response
    ↓
Validation (Pydantic)
    ↓
Citrus Backend (Ktor)
    ↓
PostgreSQL Database
```

### Design Principles

1. **Local-First**: Prioritize local LLMs for privacy and cost
2. **Provider Agnostic**: Abstract LLM providers behind unified interface
3. **Type Safety**: Full Pydantic validation throughout
4. **Stateless**: No session state, easy to scale horizontally
5. **Prompt Engineering**: Optimized system prompts for accuracy
6. **Error Handling**: Graceful degradation and clear error messages

---

## 🤝 Contributing

This is part of the Citrus Platform. For contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

Part of the [Citrus Platform](https://github.com/jlmalone/citrus-web-modern)

---

## 🔗 Related Projects

- [citrus-web-modern](https://github.com/jlmalone/citrus-web-modern) - Main Citrus web application
- [Citrus Backend](https://github.com/jlmalone/citrus-backend) - Kotlin/Ktor backend

---

## 📞 Support

For issues or questions:

- GitHub Issues: [citrus-llm-engine/issues](https://github.com/jlmalone/citrus-llm-engine/issues)
- Citrus Platform: [citrus-web-modern](https://github.com/jlmalone/citrus-web-modern)

---

**Built with FastAPI + Local LLMs** 🚀
