# Citrus LLM Engine

OpenAI-compatible receipt extraction service using local LLMs (LM Studio, Ollama) with FastAPI.

## Overview

Citrus LLM Engine is a microservice that takes OCR text from receipts and returns structured JSON data using local LLMs. It provides:

- **OpenAI-compatible API** for maximum flexibility
- **Swappable LLM providers** (LM Studio, Ollama, OpenAI)
- **Receipt-specific extraction endpoint** with optimized prompts
- **Type-safe Pydantic models** for validation
- **Docker deployment** for easy scaling

## Features

- **Local-first**: Works with LM Studio and Ollama for privacy and cost savings
- **FastAPI**: Modern, high-performance Python web framework
- **OpenAI-compatible**: Standard `/v1/chat/completions` endpoint
- **Specialized extraction**: Optimized `/api/extract` endpoint for receipts
- **Production-ready**: Docker, health checks, logging, tests

## Quick Start

### Prerequisites

1. **Python 3.11+** or **Docker**
2. One of the following LLM providers:
   - **LM Studio** (recommended for local development)
   - **Ollama**
   - **OpenAI API key** (fallback)

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd citrus-llm-engine
   ```

2. **Start LM Studio** (or Ollama)
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Load a model (e.g., `llama-3-8b-instruct`, `mistral-7b-instruct`)
   - Start the local server (default: `http://localhost:1234`)

3. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env to customize settings
   ```

4. **Start the service**
   ```bash
   docker-compose up --build
   ```

5. **Test the API**
   ```bash
   curl http://localhost:8000/api/health
   ```

### Option 2: Local Development

1. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env as needed
   ```

3. **Run the server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

## API Documentation

### Endpoints

- `GET /` - API information
- `GET /api/health` - Health check
- `POST /api/extract` - Receipt extraction (convenience endpoint)
- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `GET /docs` - Interactive API documentation (Swagger UI)

### Extract Receipt

**Endpoint**: `POST /api/extract`

**Request**:
```json
{
  "ocr_text": "WAL*MART #1234\n123 Main St\n12/25/2024 3:45 PM\n\nMILK 2% GAL    $3.99\nBREAD WHEAT    $2.49\nEGGS DOZEN     $4.29\n\nSUBTOTAL      $10.77\nTAX            $0.75\nTOTAL         $11.52",
  "model": "local-model",
  "temperature": 0.1
}
```

**Response**:
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
      "total": 11.52,
      "tip": null,
      "discount": null
    },
    "payment_method": null,
    "currency": "USD"
  },
  "error": null,
  "raw_response": "{...}"
}
```

### Chat Completions (OpenAI-compatible)

**Endpoint**: `POST /v1/chat/completions`

**Request**:
```json
{
  "model": "local-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

**Response**: Standard OpenAI format

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# LLM Provider (lmstudio, ollama, openai)
LLM_PROVIDER=lmstudio

# LM Studio Settings
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# OpenAI Settings
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Request Settings
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=2000
REQUEST_TIMEOUT=120
```

### LLM Provider Setup

#### LM Studio (Recommended)

1. Download [LM Studio](https://lmstudio.ai/)
2. Load a model (recommended: `llama-3-8b-instruct`, `mistral-7b-instruct`)
3. Start local server (Server tab → Start server)
4. Set `LLM_PROVIDER=lmstudio` in `.env`

#### Ollama

1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama3`
3. Start Ollama service
4. Set `LLM_PROVIDER=ollama` in `.env`

#### OpenAI

1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set `OPENAI_API_KEY` in `.env`
3. Set `LLM_PROVIDER=openai` in `.env`

## Testing

### Run Tests

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_extract.py
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/api/health

# Extract receipt
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_text": "WALMART #1234\nMILK $3.99\nBREAD $2.49\nTOTAL $6.48"
  }'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production Considerations

1. **Environment Variables**: Use secure secret management
2. **CORS**: Configure `allow_origins` in `app/main.py`
3. **Rate Limiting**: Add rate limiting middleware
4. **Logging**: Configure log aggregation
5. **Monitoring**: Add health check monitoring
6. **Scaling**: Use container orchestration (Kubernetes, Docker Swarm)

## Project Structure

```
citrus-llm-engine/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py       # Request Pydantic models
│   │   └── response.py      # Response Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── extract.py       # Receipt extraction endpoint
│   │   └── chat.py          # Chat completion endpoint
│   └── services/
│       ├── __init__.py
│       ├── llm_service.py   # LLM provider abstraction
│       └── prompt_builder.py # System prompt management
├── tests/
│   ├── __init__.py
│   ├── test_extract.py      # Extraction tests
│   ├── test_llm_providers.py # Provider tests
│   └── fixtures/
│       └── sample_receipts.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Development

### Code Quality

```bash
# Format code
black app/ tests/

# Lint
ruff app/ tests/

# Type check
mypy app/
```

### Adding New Endpoints

1. Create router in `app/routers/`
2. Add business logic in `app/services/`
3. Define models in `app/models/`
4. Include router in `app/main.py`
5. Add tests in `tests/`

## Integration with Citrus Backend

The Citrus Ktor backend can call this service:

```kotlin
// Kotlin example
val response = httpClient.post("http://citrus-llm-engine:8000/api/extract") {
    contentType(ContentType.Application.Json)
    setBody(ExtractionRequest(ocrText = receiptText))
}
```

## Troubleshooting

### LM Studio Connection Issues

- Ensure LM Studio server is running
- Check URL: `http://localhost:1234/v1`
- In Docker: Use `host.docker.internal:1234` (Mac/Windows) or `172.17.0.1:1234` (Linux)

### Ollama Connection Issues

- Verify Ollama is running: `ollama list`
- Check service: `curl http://localhost:11434/api/tags`
- In Docker: Use `host.docker.internal:11434`

### Extraction Quality

- Use better models (llama-3-70b, mixtral-8x7b)
- Lower temperature (0.0 - 0.2) for more consistent results
- Increase max_tokens if responses are truncated
- Fine-tune system prompt in `app/services/prompt_builder.py`

## Performance

- **LM Studio**: ~2-5s per request (8B model, CPU)
- **Ollama**: ~2-5s per request (8B model, CPU)
- **OpenAI**: ~1-2s per request (network dependent)

GPU acceleration significantly improves performance (5-10x faster).

## License

MIT

## Support

For issues and questions, please open an issue on GitHub.

---

Part of the [Citrus Platform](https://github.com/jlmalone/citrus-web-modern)
