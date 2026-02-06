# Feature: Phase 1 - Foundation Setup

> **IMPORTANT**: Validate documentation and codebase patterns before implementing. Pay attention to naming of existing utils, types, and models. Import from the right files.

## Feature Description

Set up the foundational infrastructure for the Ultimate RAG system including project structure, dependency management, database schema, and configuration system. This phase creates the scaffold upon which all other phases build.

## User Story

As a developer
I want a properly configured Python project with database connectivity
So that I can build the RAG ingestion and agent features on a solid foundation

## Problem Statement

No project structure exists yet. We need:
- `pyproject.toml` with all dependencies
- PostgreSQL + pgvector Docker container
- Database schema (4 tables)
- Pydantic configuration management

## Solution Statement

Create a modern Python project using:
- **uv** for dependency management
- **Pydantic-settings** for configuration
- **langchain-postgres** for pgvector integration
- **Docker** for PostgreSQL with pgvector

## Feature Metadata

- **Feature Type**: New Capability
- **Estimated Complexity**: Medium
- **Primary Systems Affected**: Database, Configuration
- **Dependencies**: Docker, PostgreSQL, pgvector, uv

---

## CONTEXT REFERENCES

### Relevant Codebase Files

| File | Lines | Why |
|------|-------|-----|
| `c:\Environment\RAG\.env` | 1-54 | Existing environment variables - must match |
| `c:\Environment\RAG\Lumosity-RAG\requirements.txt` | 1-9 | Reference for shared dependencies |
| `c:\Environment\RAG\Lumosity-RAG\config.py` | all | Pattern for Gemini config |

### New Files to Create

```
ultimate_rag/
├── pyproject.toml                    # Dependencies + project metadata
├── src/
│   ├── __init__.py                   # Package marker
│   ├── config.py                     # Pydantic settings
│   └── database/
│       ├── __init__.py               # Package marker
│       ├── connection.py             # Async PostgreSQL connection
│       ├── models.py                 # SQLAlchemy models (optional Phase 1)
│       └── schema.sql                # Table definitions
└── tests/
    ├── __init__.py                   # Package marker
    └── test_database.py              # Connection + schema tests
```

### Relevant Documentation

| Resource | Why |
|----------|-----|
| [langchain-postgres PGVector](https://github.com/langchain-ai/langchain-postgres) | Connection patterns |
| [pydantic-settings .env loading](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Configuration pattern |
| [pgvector Docker image](https://hub.docker.com/r/pgvector/pgvector) | Container setup |

### Patterns to Follow

**Connection String Pattern** (from langchain-postgres):
```python
CONNECTION_STRING = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
    f":{POSTGRES_PORT}/{POSTGRES_DB}"
)
```

**Pydantic Settings Pattern**:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    database_url: str
    gemini_api_key: str
```

---

## IMPLEMENTATION PLAN

### Phase 1.1: Project Scaffolding
- Create directory structure
- Create pyproject.toml with dependencies
- Initialize Python package

### Phase 1.2: Configuration System
- Create Pydantic settings class
- Load from existing .env file
- Validate all required variables

### Phase 1.3: Database Setup
- Start PostgreSQL Docker container
- Create schema.sql with 4 tables
- Implement connection module

### Phase 1.4: Testing
- Create database connection tests
- Verify tables exist
- Test basic CRUD operations

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `ultimate_rag/` directory structure

- **IMPLEMENT**: Create all directories in one go
- **COMMAND**:
```powershell
mkdir -p ultimate_rag/src/database
mkdir -p ultimate_rag/src/ingestion/processors
mkdir -p ultimate_rag/src/agent/tools
mkdir -p ultimate_rag/src/api
mkdir -p ultimate_rag/src/cleanup
mkdir -p ultimate_rag/src/sources
mkdir -p ultimate_rag/tests
```
- **VALIDATE**: `ls ultimate_rag/src` shows all subdirectories

---

### Task 2: CREATE `ultimate_rag/pyproject.toml`

- **IMPLEMENT**: Modern Python project config with uv compatibility
- **IMPORTS/DEPENDENCIES**:
```toml
[project]
name = "ultimate-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.9",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "langchain-postgres>=0.0.16",
    "psycopg[binary]>=3.1.0",
    "asyncpg>=0.29.0",
    "pgvector>=0.3.0",
    "sqlalchemy>=2.0.0",
    "google-generativeai>=0.8.0",
    "pandas>=2.2.0",
    "openpyxl>=3.1.0",
    "PyMuPDF>=1.24.0",
    "pillow>=10.0.0",
    "python-dotenv>=1.0.0",
    "pydantic-settings>=2.0.0",
    "httpx>=0.27.0",
    "apscheduler>=3.10.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```
- **GOTCHA**: Use `psycopg[binary]` not `psycopg2-binary` for async support
- **VALIDATE**: `uv pip install -e ".[dev]"` succeeds

---

### Task 3: CREATE `ultimate_rag/src/__init__.py` (and all package markers)

- **IMPLEMENT**: Empty `__init__.py` files in all packages
- **FILES**:
  - `src/__init__.py`
  - `src/database/__init__.py`
  - `src/ingestion/__init__.py`
  - `src/ingestion/processors/__init__.py`
  - `src/agent/__init__.py`
  - `src/agent/tools/__init__.py`
  - `src/api/__init__.py`
  - `src/cleanup/__init__.py`
  - `src/sources/__init__.py`
  - `tests/__init__.py`
- **VALIDATE**: `python -c "from src import database"` works

---

### Task 4: CREATE `ultimate_rag/src/config.py`

- **IMPLEMENT**: Pydantic settings loading from .env
- **PATTERN**: BaseSettings with SettingsConfigDict
- **CODE**:
```python
"""Configuration management using Pydantic Settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file="../.env",  # Parent directory 
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )
    
    # Database
    database_url: str
    postgres_user: str
    postgres_password: str
    postgres_db: str
    
    # Gemini API
    gemini_api_key: str
    gemini_chat_model: str = "models/gemini-2.5-pro"
    gemini_flash_model: str = "models/gemini-2.5-flash"
    gemini_embedding_model: str = "models/embedding-001"
    
    # Google Drive (optional for Phase 1)
    google_drive_client_id: str = ""
    google_drive_client_secret: str = ""
    google_drive_folder_id: str = ""
    google_drive_refresh_token: str = ""
    
    # Webhook Auth
    webhook_auth_header: str = "X-API-Key"
    webhook_auth_value: str = ""
    
    @property
    def async_database_url(self) -> str:
        """Convert standard URL to asyncpg format."""
        return self.database_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```
- **GOTCHA**: Use `extra="ignore"` to skip env vars we don't define
- **VALIDATE**: `python -c "from src.config import get_settings; print(get_settings().database_url)"`

---

### Task 5: START PostgreSQL + pgvector Docker container

- **IMPLEMENT**: Run pgvector/pgvector:pg17 container
- **COMMAND**:
```powershell
docker run -d `
  --name ultimate-rag-db `
  -e POSTGRES_USER=rag_user `
  -e POSTGRES_PASSWORD=rag_password `
  -e POSTGRES_DB=ultimate_rag `
  -p 5432:5432 `
  pgvector/pgvector:pg17
```
- **GOTCHA**: If port 5432 is in use, stop existing containers first
- **VALIDATE**: `docker ps | Select-String ultimate-rag-db`

---

### Task 6: CREATE `ultimate_rag/src/database/schema.sql`

- **IMPLEMENT**: All 4 tables matching n8n workflow exactly
- **CODE**:
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Vector store table (matches n8n's documents_pg)
CREATE TABLE IF NOT EXISTS documents_pg (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(768)  -- Gemini embedding dimension
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS documents_pg_embedding_idx 
ON documents_pg USING hnsw (embedding vector_cosine_ops);

-- Document metadata table
CREATE TABLE IF NOT EXISTS document_metadata (
    id TEXT PRIMARY KEY,
    title TEXT,
    url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    schema TEXT  -- Column schema for tabular files
);

-- Tabular data storage (rows from CSV/XLSX)
CREATE TABLE IF NOT EXISTS document_rows (
    id SERIAL PRIMARY KEY,
    dataset_id TEXT REFERENCES document_metadata(id) ON DELETE CASCADE,
    row_data JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS document_rows_dataset_idx ON document_rows(dataset_id);

-- Chat memory table
CREATE TABLE IF NOT EXISTS chat_memory (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('human', 'ai')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_memory_session_idx ON chat_memory(session_id);
```
- **VALIDATE**: File exists and SQL syntax is valid

---

### Task 7: CREATE `ultimate_rag/src/database/connection.py`

- **IMPLEMENT**: Async PostgreSQL connection using asyncpg
- **PATTERN**: Connection pool with context manager
- **CODE**:
```python
"""PostgreSQL database connection management."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import asyncpg
from asyncpg import Pool, Connection

from src.config import get_settings


_pool: Pool | None = None


async def get_pool() -> Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            host="localhost",
            port=5432,
            min_size=2,
            max_size=10,
        )
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection, None]:
    """Get a database connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_schema() -> None:
    """Initialize database schema from schema.sql."""
    schema_path = Path(__file__).parent / "schema.sql"
    schema_sql = schema_path.read_text()
    
    async with get_connection() as conn:
        await conn.execute(schema_sql)


async def check_tables_exist() -> dict[str, bool]:
    """Check if all required tables exist."""
    tables = ["documents_pg", "document_metadata", "document_rows", "chat_memory"]
    result = {}
    
    async with get_connection() as conn:
        for table in tables:
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
                """,
                table
            )
            result[table] = exists
    
    return result
```
- **GOTCHA**: asyncpg uses `$1` placeholders, not `%s`
- **VALIDATE**: See Task 9 tests

---

### Task 8: CREATE `ultimate_rag/src/database/__init__.py` with exports

- **IMPLEMENT**: Export public API from database module
- **CODE**:
```python
"""Database module exports."""
from .connection import (
    get_pool,
    get_connection,
    close_pool,
    init_schema,
    check_tables_exist,
)

__all__ = [
    "get_pool",
    "get_connection", 
    "close_pool",
    "init_schema",
    "check_tables_exist",
]
```
- **VALIDATE**: `python -c "from src.database import init_schema"`

---

### Task 9: CREATE `ultimate_rag/tests/test_database.py`

- **IMPLEMENT**: Async tests for database connectivity and schema
- **PATTERN**: pytest-asyncio with fixtures
- **CODE**:
```python
"""Database connection and schema tests."""
import pytest
import pytest_asyncio

from src.database import (
    get_pool,
    get_connection,
    close_pool,
    init_schema,
    check_tables_exist,
)


@pytest_asyncio.fixture(autouse=True)
async def setup_teardown():
    """Setup and teardown for each test."""
    yield
    await close_pool()


@pytest.mark.asyncio
async def test_connection_pool():
    """Test that we can create a connection pool."""
    pool = await get_pool()
    assert pool is not None
    assert pool.get_size() >= 1


@pytest.mark.asyncio
async def test_get_connection():
    """Test that we can acquire a connection."""
    async with get_connection() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1


@pytest.mark.asyncio
async def test_init_schema():
    """Test schema initialization creates all tables."""
    await init_schema()
    tables = await check_tables_exist()
    
    assert tables["documents_pg"] is True
    assert tables["document_metadata"] is True
    assert tables["document_rows"] is True
    assert tables["chat_memory"] is True


@pytest.mark.asyncio
async def test_pgvector_extension():
    """Test that pgvector extension is enabled."""
    async with get_connection() as conn:
        result = await conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        )
        assert result == "vector"


@pytest.mark.asyncio 
async def test_documents_pg_structure():
    """Test documents_pg table has correct columns."""
    await init_schema()
    
    async with get_connection() as conn:
        # Insert a test document (without embedding)
        await conn.execute(
            """
            INSERT INTO documents_pg (content, metadata)
            VALUES ($1, $2)
            """,
            "Test content",
            '{"file_id": "test123"}'
        )
        
        # Verify we can query it
        row = await conn.fetchrow(
            "SELECT * FROM documents_pg WHERE content = $1",
            "Test content"
        )
        assert row["content"] == "Test content"
        assert row["metadata"]["file_id"] == "test123"
        
        # Cleanup
        await conn.execute("DELETE FROM documents_pg WHERE content = $1", "Test content")
```
- **VALIDATE**: `cd ultimate_rag && pytest tests/test_database.py -v`

---

### Task 10: UPDATE `.env` to add missing GEMINI_FLASH_MODEL

- **IMPLEMENT**: Add flash model config that was identified during planning
- **LOCATION**: `c:\Environment\RAG\.env` line 53
- **ADD**:
```
GEMINI_FLASH_MODEL=models/gemini-2.5-flash
```
- **VALIDATE**: `grep GEMINI_FLASH .env`

---

## TESTING STRATEGY

### Unit Tests (pytest)
- Test database connection pool creation
- Test individual table existence
- Test pgvector extension activation
- Test config loading from .env

### Integration Tests
- Full schema initialization
- Insert and query operations
- Vector embedding storage (placeholder)

### Edge Cases
- Missing .env file → clear error message
- Database not running → connection error handling
- Invalid credentials → authentication error

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
```powershell
cd ultimate_rag
python -m py_compile src/config.py
python -m py_compile src/database/connection.py
```

### Level 2: Configuration
```powershell
cd ultimate_rag
python -c "from src.config import get_settings; s = get_settings(); print(f'DB: {s.database_url}')"
```

### Level 3: Database Connection
```powershell
docker ps | Select-String ultimate-rag-db
```

### Level 4: Unit Tests
```powershell
cd ultimate_rag
pytest tests/test_database.py -v
```

### Level 5: Manual Verification
```powershell
# Connect directly to check tables
docker exec -it ultimate-rag-db psql -U rag_user -d ultimate_rag -c "\dt"
```

---

## ACCEPTANCE CRITERIA

- [ ] `ultimate_rag/` directory structure created
- [ ] `pyproject.toml` with all dependencies defined
- [ ] Dependencies installed successfully (`uv pip install -e ".[dev]"`)
- [ ] `config.py` loads all .env variables
- [ ] PostgreSQL Docker container running
- [ ] pgvector extension activated
- [ ] All 4 tables created (documents_pg, document_metadata, document_rows, chat_memory)
- [ ] `pytest tests/test_database.py` passes (5/5 tests)
- [ ] No linting errors or type issues

---

## COMPLETION CHECKLIST

- [ ] All 10 tasks completed in order
- [ ] Each task validation passed immediately
- [ ] Docker container healthy
- [ ] Full test suite passes
- [ ] Configuration verified with real .env values
- [ ] Ready for Phase 2: Ingestion Pipeline

---

## NOTES

- **psycopg vs asyncpg**: Using asyncpg directly for connection pooling as it's more performant. langchain-postgres uses asyncpg internally.
- **Schema first**: Creating schema.sql separately allows manual inspection and direct SQL debugging.
- **Embedding dimension**: 768 matches Gemini's default embedding dimension.
- **HNSW index**: Using cosine operator for similarity search (standard for text embeddings).
