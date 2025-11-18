# 🤖 CLAUDE CODE WEB AGENT TASK - citrus-llm-engine

**Repository**: citrus-llm-engine
**Priority**: 🔴 CRITICAL
**Estimated Effort**: 8-12 hours
**Status**: ⚪ Pending

---

## ⚠️ IDEMPOTENCY CHECK - READ THIS FIRST!

**Before starting any work**, check if this task has already been completed:

```bash
# Check for completion marker
if [ -f ".claude-agent-completed" ]; then
    echo "✅ This task was already completed!"
    cat .claude-agent-completed
    exit 0
fi
```

**If the file exists**, this means the task is done. Display the contents and **STOP**. Do not proceed.

**If the file does NOT exist**, proceed with the task below.

---

## 📋 TASK: Create citrus-llm-engine — OpenAI-Compatible Receipt Extraction Service

### Project Overview
You are creating **citrus-llm-engine**, a microservice that takes OCR text from receipts and returns structured JSON using local LLMs (LM Studio, Ollama, etc.) with an OpenAI-compatible interface.

### Context
This is part of the Citrus receipt processing ecosystem. The backend (Ktor) will call this service via HTTP to extract structured data from receipt text.

---

## 🎯 DELIVERABLES CHECKLIST

Create these files/components:

### Core Application
- [ ] `app/main.py` - FastAPI application entry point
- [ ] `app/routers/extract.py` - `/api/extract` endpoint
- [ ] `app/routers/chat.py` - `/v1/chat/completions` (OpenAI compatible)
- [ ] `app/services/llm_service.py` - LLM provider abstraction (LM Studio/Ollama/OpenAI)
- [ ] `app/services/prompt_builder.py` - System prompt management
- [ ] `app/models/request.py` - Pydantic request models
- [ ] `app/models/response.py` - Pydantic response models
- [ ] `app/config.py` - Environment configuration

### Testing
- [ ] `tests/test_extract.py` - Extraction endpoint tests
- [ ] `tests/test_llm_providers.py` - LLM provider tests
- [ ] `tests/fixtures/sample_receipts.json` - Test fixtures

### Deployment
- [ ] `Dockerfile` - Container definition
- [ ] `docker-compose.yml` - Easy deployment setup
- [ ] `requirements.txt` - Python dependencies
- [ ] `.env.example` - Environment variables template

### Documentation
- [ ] `README.md` - Setup instructions, API docs, examples
- [ ] `pyproject.toml` - Python project metadata

---

## 📐 TECHNICAL SPECIFICATIONS

### 1. Technology Stack
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **LLM Interface**: OpenAI Python SDK (for compatibility)
- **Deployment**: Docker + docker-compose

### 2. API Endpoints

#### POST /api/extract
**Purpose**: Convenience wrapper for receipt extraction

**Request**:
```json
{
  "ocr_text": "WAL*MART #1234\n123 Main St\n12/25/2024 3:45 PM\n\nMILK 2% GAL    $3.99\nBREAD WHEAT    $2.49\nEGGS DOZEN     $4.29\n\nSUBTOTAL      $10.77\nTAX            $0.75\nTOTAL         $11.52",
  "image_url": "optional-for-vision-models",
  "model": "lmstudio-local"
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

#### POST /v1/chat/completions
**Purpose**: OpenAI-compatible interface for maximum flexibility

Standard OpenAI chat completions format.

### 3. LLM Provider Support (Swappable)

Configure via `LLM_PROVIDER` environment variable:
- `lmstudio` (default) - Local LM Studio at `http://localhost:1234/v1`
- `ollama` - Local Ollama at `http://localhost:11434`
- `openai` - OpenAI API (fallback for production)

### 4. System Prompt Engineering

Create a prompt that instructs the LLM to:
- Extract receipt data into structured JSON
- Return ONLY valid JSON (no markdown fences)
- Handle missing/unclear data gracefully
- Normalize merchant names (WAL*MART → Walmart)
- Infer item categories (MILK → Groceries > Dairy)

Example system prompt:
```
You are a receipt data extraction specialist. Given OCR text from a receipt, extract structured information in JSON format.

Return ONLY valid JSON with this exact structure:
{
  "merchant": {"name": string, "store_number": string|null, "address": string|null},
  "timestamp": ISO8601 string|null,
  "items": [{"description": string, "price": number, "quantity": number, "category": string}],
  "totals": {"subtotal": number, "tax": number, "total": number},
  "payment_method": string|null,
  "currency": string
}

Rules:
- Normalize merchant names
- Infer categories intelligently
- Handle missing data gracefully
- Never include markdown fences
```

### 5. Code Quality Requirements
- Type hints throughout (Python 3.11+)
- Pydantic models for validation
- Proper error handling
- Logging (INFO level)
- Environment-based configuration
- No hardcoded secrets

---

## 🧪 TESTING REQUIREMENTS

Include in README:
```bash
# Start LM Studio locally and load a model (e.g., llama-3-8b-instruct)

# Run the service
docker-compose up

# Test extraction
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_text": "WALMART #1234\nMILK $3.99\nBREAD $2.49\nTOTAL $6.48"
  }'
```

---

## ✅ SUCCESS CRITERIA

Before marking as complete, verify:

- [ ] Service starts and responds to health checks
- [ ] `/api/extract` endpoint extracts structured data from sample receipts
- [ ] Works with all 3 LLM providers (LM Studio, Ollama, OpenAI)
- [ ] Returns consistent JSON schema
- [ ] Unit tests pass
- [ ] Docker deployment works
- [ ] README has complete setup instructions
- [ ] `.env.example` has all configuration options

---

## 🎬 COMPLETION PROTOCOL

**When ALL tasks are complete**:

1. **Create completion marker**:
```bash
cat > .claude-agent-completed << EOF
✅ TASK COMPLETED

Repository: citrus-llm-engine
Task: Create LLM-powered receipt extraction service
Completed: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Agent: Claude Code Web

Summary:
- ✅ FastAPI application with /api/extract endpoint
- ✅ OpenAI-compatible /v1/chat/completions endpoint
- ✅ Swappable LLM providers (LM Studio, Ollama, OpenAI)
- ✅ Pydantic models and type safety
- ✅ Unit tests with sample fixtures
- ✅ Docker deployment ready
- ✅ Comprehensive README

Files Created: $(find . -type f -name "*.py" | wc -l) Python files
Tests Passing: $(pytest --co -q 2>/dev/null | tail -1 || echo "See test output")

Next Steps:
1. Start the service: docker-compose up
2. Test extraction: curl -X POST http://localhost:8000/api/extract ...
3. Integrate with Citrus backend (Ktor)
EOF
```

2. **Commit and push**:
```bash
git add .
git commit -m "🧠 Complete LLM extraction engine

- FastAPI service with OpenAI-compatible interface
- Swappable LLM providers (LM Studio/Ollama/OpenAI)
- Structured receipt extraction from OCR text
- Full test suite and Docker deployment
- Ready for integration with Citrus backend"

git push origin main
```

3. **Report completion**:
Display the contents of `.claude-agent-completed` to the user and confirm the task is done.

---

## 📝 NOTES

- Focus on **local-first** (LM Studio/Ollama), OpenAI is fallback only
- Optimize prompts for **accuracy over speed**
- Handle **messy OCR** gracefully (typos, missing data)
- This service will be called by the Ktor backend
- Design for **horizontal scaling** (stateless)

---

**Ready to build? Check for `.claude-agent-completed` first, then proceed! 🚀**
