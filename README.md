# Ultimate RAG

> **Production-Grade Retrieval Augmented Generation System**
> *Replicating the robustness of n8n workflows with the flexibility of Python.*

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Overview

**Ultimate RAG** is a modular, async-first Python library designed for building high-performance RAG applications. It acts as a bridge between your documents (PDF, CSV, JSON, etc.) and Large Language Models (Gemini, OpenAI), providing a robust pipeline for ingestion, embedding, storage, and semantic retrieval.

Key philosophy: **"Library first, Framework second."**
You can use `ultimate-rag` as a standalone library in your own apps, or run it as a service.

## ✨ Key Features

- **🔌 Modular Architecture**: Clean separation of Ingestion, Storage, and Retrieval layers.
- **📄 Multi-Format Support**: Native handling for PDF, CSV, JSON, Excel, TXT, Markdown, Images, and more.
- **🧠 Agentic Retrieval**: Uses LangChain agents with memory to "chat" with your documents, not just search them.
- **⚡ Async First**: Built on `asyncio` and `asyncpg` for high-concurrency performance.
- **🛡️ ACID Transactions**: Robust database management with PostgreSQL (pgvector).
- **🧹 Auto-Cleanup**: Built-in utilities to detect duplicates and clean orphan vector chunks.
- **📦 Zero-Config Client**: `RAGClient` facade handles connection pooling and session management automatically.

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ with `pgvector` extension
- Google Gemini API Key

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ultimate-rag.git
   cd ultimate-rag
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Configure Environment:**
   Create a `.env` file in the root directory:
   ```ini
   # Database
   DATABASE_URL=postgresql://user:pass@localhost:5432/rag_db
   POSTGRES_USER=user
   POSTGRES_PASSWORD=pass
   POSTGRES_DB=rag_db

   # AI Provider
   GEMINI_API_KEY=your_gemini_api_key
   GEMINI_CHAT_MODEL=models/gemini-2.5-pro
   GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
   ```

---

## 💡 Usage

The recommended way to use Ultimate RAG is via the **`RAGClient`** facade.

### 1. Basic Ingestion & Query

```python
import asyncio
from ultimate_rag import RAGClient

async def main():
    async with RAGClient() as client:
        # Ingest a document
        doc_id = await client.ingest("contracts/Agreement_2024.pdf")
        print(f"Ingested document: {doc_id}")

        # Chat with your data
        response = await client.query("What are the payment terms in the agreement?")
        print(f"AI: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Batch Ingestion

Efficiently process multiple files concurrently:

```python
files = ["data/report_q1.pdf", "data/report_q2.pdf", "data/financials.xlsx"]
doc_ids = await client.ingest_batch(files)
```

### 3. Direct Vector Search

Bypass the agent for raw semantic search results:

```python
# Returns raw text chunks relevant to the query
results = await client.search("revenue growth 2024", limit=5)
print(results)
```

### 4. Database Management

Manage your knowledge base programmatically:

```python
# Check stats
stats = await client.get_stats()
print(stats)

# Remove a specific document
await client.delete_document(doc_id)

# Auto-clean duplicates and orphan chunks
await client.cleanup()
```

---

## 🏗️ Architecture

### 1. High-Level Overview

```mermaid
graph LR
    User[User / App] --> Client[RAGClient Library]
    
    subgraph RAG_System
        Client --> Ingest[Ingestion Pipeline]
        Client --> Agent[RAG Agent]
        
        Ingest --> DB[(PostgreSQL + pgvector)]
        Agent --> DB
    end

    Ingest --> Gemini[Google Gemini API]
    Agent --> Gemini
```

### 2. Ingestion Workflow
*How documents are processed and stored.*

```mermaid
graph LR
    File[Document File] --> Detect[Type Detection]
    Detect --> Process[File Processor]
    Process --> Chunk[Text Chunker]
    Chunk --> Embed[Embedding Model]
    Embed --> Store[Store in DB]

    subgraph Storage
        Store --> Vectors[(Vector Chunks)]
        Store --> Meta[(Metadata)]
        Store --> Rows[(Table Rows)]
    end
```

### 3. Retrieval & Agent Workflow
*How the system answers questions.*

```mermaid
graph LR
    Query[User Query] --> Agent[RAG Agent]
    
    subgraph Agent_Tools
        Agent --> Memory[(Chat Memory)]
        Agent --> Search[Vector Search]
        Agent --> SQL[SQL Query]
    end

    Search --> Vectors[(Vector Store)]
    SQL --> Rows[(Tabular Data)]
    
    Vectors --> Context[Retrieved Context]
    Rows --> Context
    
    Context --> Synthesis[LLM Answer Generation]
    Synthesis --> Response[Final Answer]
```

### 1. Ingestion Layer (`src.ingestion`)
- **Detection**: Automatically identifies file types by extension and magic numbers.
- **Processing**: Specialized processors for each format (e.g., `PDFProcessor` uses PyMuPDF, `SpreadsheetProcessor` handles CSV/XLSX).
- **Chunking**: Intelligent text splitting ensuring context preservation.
- **Embedding**: Generates vector embeddings using Gemini models.

### 2. Storage Layer (`src.database`)
- **PostgreSQL**: Primary data store.
- **pgvector**: Handles vector similarity search.
- **Tables**:
  - `document_metadata`: File info (filename, type, dates).
  - `document_rows`: Raw tabular data (for CSV/XLSX).
  - `documents_pg`: Vector embeddings and text chunks.
  - `chat_memory`: User conversation history.

### 3. Retrieval Layer (`src.agent`)
- **LangChain Agent**: Orchestrates tools to answer user queries.
- **Tools**:
  - `vector_search`: Semantic search over embeddings.
  - `list_docs`: Lists available files.
  - `get_content`: Retrieves full document content.
- **Memory**: Persistent chat history per session.

---

## 🧪 Testing

We use `pytest` for robust testing.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_client.py

# Run with output
pytest -v
```

### Test Coverage
- **Unit Tests**: Cover individual processors and database functions.
- **Integration Tests**: Verify `RAGClient` workflows (Ingest -> Query).
- **Live Verification**: `tests/test_verify_client_live.py` runs against a real database instance to ensure end-to-end correctness.

---

## 🗺️ Roadmap

- [x] **Phase 1**: Foundation & Database Cleanup
- [x] **Phase 2**: Universal Ingestion Pipeline
- [x] **Phase 3**: RAG Agent & Memory
- [x] **Phase 3.5**: Advanced DB Management
- [x] **Phase 4**: Modular `RAGClient` Library
- [ ] **Phase 5**: CLI Tool (`rag --ingest file.pdf`)
- [ ] **Phase 6**: REST API Server (FastAPI)

---

## 🤝 Integration

To integrate `ultimate-rag` into your project:

1. **Install** `ultimate-rag` in your environment.
2. **Import** `RAGClient`.
3. **Session Management**: Use a unique `session_id` for each user session.
4. **Cleanup**: Call `client.delete_document(id)` or `client.cleanup()` after the user session ends to maintain ephemeral data privacy.

```python
# Example: Ephemeral Session
session_id = f"user_{user_id}_{timestamp}"
async with RAGClient(session_id=session_id) as rag:
    try:
        await rag.ingest(user_file)
        answer = await rag.query(user_question)
        return answer
    finally:
        # cleanup logic here
        pass
```

---
