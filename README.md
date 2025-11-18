# 🧠 citrus-llm-engine

LLM-powered receipt extraction service with OpenAI-compatible interface.

## Status

⚪ **Not Started** - See `AGENT_PROMPT.md` for build instructions

## Quick Start

This repository contains a Claude Code Web agent prompt. To build this project:

1. Open this repository in Claude Code Web
2. Copy the contents of `AGENT_PROMPT.md`
3. Paste into the chat
4. The agent will build everything automatically

The agent will create a `.claude-agent-completed` marker when done.

## What This Does

Converts OCR text from receipts into structured JSON using local LLMs:
- LM Studio (local)
- Ollama (local)
- OpenAI (fallback)

## Architecture

```
Receipt OCR Text → LLM Engine → Structured JSON
```

Part of the [Citrus Platform](https://github.com/jlmalone/citrus-web-modern)
