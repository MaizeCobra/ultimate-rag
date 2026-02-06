# Ultimate RAG System - Product Requirements Document

## 1. Executive Summary

**Ultimate RAG** is a Python-based Retrieval-Augmented Generation system designed to provide intelligent document search and chat capabilities across multimodal content. The system converts an existing n8n workflow into a standalone, production-ready Python application.

The core value proposition is enabling users to interact with their document corpus through natural language, leveraging AI to search, analyze, and extract insights from text, spreadsheets, images, audio, video, and PDFs. The system provides hybrid search capabilities combining vector similarity with SQL queries for tabular data.

**MVP Goal:** Achieve 100% feature parity with the n8n "Ultimate RAG" workflow, deployed as a self-contained Python service with Google Drive integration and PostgreSQL vector storage.

---

## 2. Mission

**Mission Statement:** Democratize intelligent document access by providing a powerful, self-hosted RAG system that understands and retrieves information from any document format.

### Core Principles

1. **Feature Parity First** - Match n8n workflow exactly before extending
2. **Multimodal Native** - Treat all content types as first-class citizens
3. **Modular Architecture** - Swappable components (sources, rerankers)
4. **Developer Experience** - Simple setup with Docker + environment variables
5. **AI-First Search** - Let the agent decide the best retrieval strategy

---

## 3. Target Users

### Primary Persona: Knowledge Worker

- **Role:** Researcher, analyst, or professional managing document libraries
- **Technical Comfort:** Comfortable with basic CLI, Docker, and API calls
- **Pain Points:**
  - Manual document searching across multiple file types
  - Inability to query tabular data with natural language
  - Loss of context from audio/video content

### Secondary Persona: Developer/Integrator

- **Role:** Building applications on top of the RAG system
- **Technical Comfort:** High - writes code, manages APIs
- **Pain Points:**
  - n8n workflow lock-in
  - Need for programmatic access to RAG capabilities

---

## 4. MVP Scope

### In Scope (Core Functionality)

- ✅ Ingest 8 file types: TXT, JSON, CSV, XLSX, Image, Audio, Video, PDF
- ✅ Vector storage with PostgreSQL + pgvector
- ✅ AI agent with 5 tools (RAG, List Docs, Get Content, SQL Query, Re-ranker)
- ✅ Chat interface with session-based memory
- ✅ Google Drive file watching (create/update triggers)
- ✅ 15-minute cleanup job for trashed files
- ✅ Webhook endpoint with header authentication

### In Scope (Technical)

- ✅ LangChain framework integration
- ✅ Gemini API (2.5-pro for chat/video, 2.5-flash for image/audio/PDF)
- ✅ Docker-based PostgreSQL with pgvector
- ✅ FastAPI REST endpoints

### Out of Scope (Phase 2+)

- ❌ Google Discovery Engine re-ranking (requires GCP Service Account)
- ❌ Real-time Google Drive push notifications (polling only for MVP)
- ❌ Multi-tenant support
- ❌ Web UI for chat (API only for MVP)
- ❌ Streaming responses

---

## 5. User Stories

### Primary Stories

1. **As a knowledge worker**, I want to ask questions about my documents in natural language, so that I can find information without manual searching.
   - *Example:* "What is the total price of all chocolate cakes in the catalog?"

2. **As a knowledge worker**, I want to upload PDFs with images, so that the system understands visual content too.
   - *Example:* Upload product catalog PDF → AI describes product images

3. **As a knowledge worker**, I want to query spreadsheet data with natural language, so that I don't need to write SQL.
   - *Example:* "Show me all products under ₹500 that are eggless"

4. **As a developer**, I want to trigger document ingestion via webhook, so that I can integrate with other systems.
   - *Example:* POST to `/webhook` with file path → automatic ingestion

5. **As a knowledge worker**, I want the system to monitor my Google Drive folder, so that new documents are automatically indexed.
   - *Example:* Drop file in Drive folder → appears in RAG within minutes

6. **As a developer**, I want session-based chat memory, so that users can have contextual conversations.
   - *Example:* Follow-up questions reference previous answers

### Technical Stories

7. **As an operator**, I want automatic cleanup of deleted files, so that my vector store stays clean.
   - *Example:* Trash file in Drive → removed from database within 15 minutes

8. **As a developer**, I want modular file source integration, so that I can swap Google Drive for S3 later.
   - *Example:* Implement `FileSource` interface for any storage backend

---

## 6. Core Architecture & Patterns

### High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Google Drive   │────▶│  Ingestion       │────▶│   PostgreSQL    │
│  (or API Upload)│     │  Pipeline        │     │   + pgvector    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
┌─────────────────┐     ┌──────────────────┐             │
│    FastAPI      │◀───▶│    AI Agent      │◀────────────┘
│    Endpoints    │     │  (LangChain)     │
└─────────────────┘     └──────────────────┘
```

### Directory Structure

```
ultimate_rag/
├── src/
│   ├── config.py           # Pydantic settings
│   ├── main.py             # FastAPI app
│   ├── database/           # PostgreSQL + models
│   ├── ingestion/          # File processors
│   │   └── processors/     # Per-type handlers
│   ├── agent/              # LangChain agent
│   │   ├── tools/          # 5 AI tools
│   │   └── memory.py       # PostgreSQL memory
│   ├── sources/            # File source adapters
│   ├── api/                # Routes + auth
│   └── cleanup/            # Scheduled jobs
└── tests/
```

### Key Patterns

| Pattern | Usage |
|---------|-------|
| **Strategy** | File processors, rerankers |
| **Factory** | Tool creation based on config |
| **Repository** | Database access abstraction |
| **Dependency Injection** | FastAPI dependencies |

---

## 7. Tools/Features

### AI Agent Tools

| Tool | Purpose | Key Operations |
|------|---------|----------------|
| **RAG Tool** | Vector similarity search | Query pgvector, return top 25 chunks |
| **List Documents** | Browse available files | SELECT from document_metadata |
| **Get File Contents** | Retrieve full document | Aggregate chunks by file_id |
| **SQL Query** | Query tabular data | AI-generated SQL on document_rows |
| **Re-ranker** | Improve search quality | Google Discovery Engine API (Phase 2) |

### Ingestion Features

| File Type | Processor | Gemini Model |
|-----------|-----------|--------------|
| Text/JSON | Direct read | - |
| CSV/XLSX | pandas → JSONB rows | - |
| Image | Vision analysis | gemini-2.5-flash |
| Audio | Transcription | gemini-2.5-flash |
| Video | Analysis + description | gemini-2.5-pro |
| PDF | Lumosity-RAG + batching | gemini-2.5-flash |

---

## 8. Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | ≥0.115.0 | API framework |
| `langchain` | ≥0.3.0 | Agent + RAG |
| `langchain-google-genai` | ≥4.2.0 | Gemini integration |
| `langchain-postgres` | ≥0.0.16 | pgvector store |
| `google-generativeai` | ≥0.8.0 | Direct Gemini API |
| `psycopg2-binary` | ≥2.9.9 | PostgreSQL driver |
| `PyMuPDF` | ≥1.24.0 | PDF extraction |
| `pandas` | ≥2.2.0 | Spreadsheet processing |
| `apscheduler` | ≥3.10.0 | Cleanup scheduler |

### Infrastructure

| Component | Implementation |
|-----------|----------------|
| Database | PostgreSQL 17 + pgvector |
| Container | Docker (`pgvector/pgvector:pg17`) |
| Python | 3.11+ |

---

## 9. Security & Configuration

### Authentication

| Endpoint | Auth Method |
|----------|-------------|
| `/chat` | None (or session token) |
| `/webhook` | Header-based (`X-API-Key`) |
| `/upload` | Header-based (`X-API-Key`) |

### Environment Variables

```bash
# Required
DATABASE_URL=postgresql://...
GEMINI_API_KEY=...

# Google Drive (Free)
GOOGLE_DRIVE_CLIENT_ID=...
GOOGLE_DRIVE_CLIENT_SECRET=...
GOOGLE_DRIVE_FOLDER_ID=...

# Optional (Phase 2)
GCP_PROJECT_ID=...
GOOGLE_APPLICATION_CREDENTIALS=...
```

### Security Scope

- ✅ Webhook authentication via shared secret
- ✅ Database credentials in environment vars
- ❌ Rate limiting (out of scope for MVP)
- ❌ HTTPS termination (handled by reverse proxy)

---

## 10. API Specification

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send message to AI agent |
| POST | `/webhook` | Trigger file ingestion |
| POST | `/upload` | Upload file directly |
| GET | `/documents` | List all documents |
| GET | `/health` | Health check |

### Example: Chat Request

```json
POST /chat
{
  "message": "What documents do I have?",
  "session_id": "user-123"
}
```

### Example: Chat Response

```json
{
  "response": "You have 5 documents in your knowledge base...",
  "session_id": "user-123",
  "tools_used": ["list_documents"]
}
```

---

## 11. Success Criteria

### MVP Success Definition

The system is considered successful when it can:
1. Ingest all 8 supported file types
2. Answer questions using appropriate tools
3. Maintain conversation memory across sessions
4. Automatically sync with Google Drive

### Functional Requirements

- ✅ Process 100-page PDF in under 5 minutes
- ✅ Return RAG results in under 3 seconds
- ✅ Handle concurrent chat sessions
- ✅ Clean up trashed files within 15 minutes

### Quality Indicators

- Test coverage ≥80% for core modules
- Zero critical security vulnerabilities
- Clear error messages for all failure modes

---

## 12. Implementation Phases

### Phase 1: Foundation (Days 1-2)

**Goal:** Database + project structure

**Deliverables:**
- ✅ Project scaffolding with pyproject.toml
- ✅ PostgreSQL + pgvector Docker setup
- ✅ Database schema (4 tables)
- ✅ Config management with Pydantic

**Validation:** `pytest tests/test_database.py` passes

---

### Phase 2: Ingestion Pipeline (Days 3-5)

**Goal:** Process all file types

**Deliverables:**
- ✅ File type detection (MIME → category)
- ✅ Text/JSON processor
- ✅ CSV/XLSX processor with row storage
- ✅ Image/Audio/Video processors (Gemini)
- ✅ PDF processor (Lumosity-RAG integration)
- ✅ Text chunking (1000 chars, 200 overlap)
- ✅ Embedding generation

**Validation:** Upload each file type → verify in database

---

### Phase 3: AI Agent (Days 6-8)

**Goal:** Working chat with tools

**Deliverables:**
- ✅ RAG tool (vector search)
- ✅ List Documents tool
- ✅ Get File Contents tool
- ✅ SQL Query tool
- ✅ Agent with system prompt
- ✅ PostgreSQL chat memory

**Validation:** Natural language queries return correct results

---

### Phase 4: Integration (Days 9-10)

**Goal:** Complete system

**Deliverables:**
- ✅ FastAPI endpoints
- ✅ Webhook authentication
- ✅ Google Drive file watcher
- ✅ Cleanup scheduler (15-min)
- ✅ End-to-end testing

**Validation:** Full workflow from Drive upload to chat query

---

## 13. Future Considerations

### Phase 2 Enhancements

- **Re-ranker Integration** - Google Discovery Engine with GCP Service Account
- **Streaming Responses** - Real-time token streaming for chat
- **Web UI** - React/Vue frontend for non-technical users

### Phase 3+ Ideas

- Multi-tenant isolation
- Custom embedding models
- Hybrid search (keyword + vector)
- Document versioning
- Analytics dashboard

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Gemini rate limits** | High | Implement exponential backoff, batch requests |
| **Large PDF processing** | Medium | 3-page batching, async processing queue |
| **pgvector performance** | Medium | HNSW index, connection pooling |
| **Google Drive API quotas** | Low | Polling interval of 5 minutes, quota monitoring |
| **OAuth token expiry** | Medium | Auto-refresh mechanism in Drive client |

---

## 15. Appendix

### Related Documents

- [Implementation Plan](file:///C:/Users/rohan_pmxdc93/.gemini/antigravity/brain/fbd9bd72-979f-4772-aa2b-c2d76faf0067/implementation_plan.md)
- [Verification Checklist](file:///C:/Users/rohan_pmxdc93/.gemini/antigravity/brain/fbd9bd72-979f-4772-aa2b-c2d76faf0067/verification_checklist.md)

### Key Dependencies

| Repository | Purpose |
|------------|---------|
| [Lumosity-RAG](file:///c:/Environment/RAG/Lumosity-RAG) | PDF extraction + vision |
| [n8n Ultimate RAG](file:///c:/Environment/RAG/Ultimate%20RAG.json) | Reference workflow |
| [pgvector](https://github.com/pgvector/pgvector) | Vector extension |

### Model Reference

| Use Case | Model | Rationale |
|----------|-------|-----------|
| Chat | gemini-2.5-pro | Highest quality reasoning |
| Video | gemini-2.5-pro | Complex temporal analysis |
| Image | gemini-2.5-flash | Fast, cost-effective |
| Audio | gemini-2.5-flash | Transcription doesn't need pro |
| PDF | gemini-2.5-flash | Matches n8n exactly |
| Embeddings | embedding-001 | Default Gemini embeddings |
