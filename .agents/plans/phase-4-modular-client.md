# Feature: Phase 4 - Modular RAG Client

The following plan should be complete, but validate documentation and codebase patterns before implementing.

## Feature Description

Implement `RAGClient`, a high-level, production-ready Python facade that transforms the RAG system into a reusable library. This client will encapsulate ingestion, querying, and database management, providing a centralized API for external applications (e.g., CLI tools, web servers, other Python scripts).

## User Story

As a developer integrating RAG into a larger application
I want to check `RAGClient` as a context manager (`async with`)
So that I can safely manage database connections and resources without manual cleanup.

## Problem Statement

Current system is a collection of loose scripts and modules.
- **No central entry point**: Users must know internal module structure (`src.agent`, `src.ingestion`).
- **Resource leaks**: Manually calling `close_pool()` is error-prone.
- **Complex setup**: setting up agent, memory, and database requires multiple steps.
- **No direct vector search**: Vector search is buried inside the agent tools.

## Solution Statement

Create `src/client.py` class `RAGClient` that:
1.  Implements Async Context Manager protocol (`__aenter__`, `__aexit__`).
2.  Wraps all core functionality (Ingest, Query, Manage, Search).
3.  Handles configuration and resource lifecycle automatically.

## Feature Metadata

**Feature Type**: Refactor / New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: `src/client.py`, `src/__init__.py`
**Dependencies**: `asyncpg`, `langchain`, `src.config`

---

## CONTEXT REFERENCES

### Relevant Codebase Files

| File | Why |
|------|-----|
| [connection.py](file:///c:/Environment/RAG/ultimate_rag/src/database/connection.py) | `get_pool`, `close_pool`. Client must manage this lifecycle. |
| [agent.py](file:///c:/Environment/RAG/ultimate_rag/src/agent/agent.py) | `RAGAgent` class. Client instantiates and delegates to this. |
| [pipeline.py](file:///c:/Environment/RAG/ultimate_rag/src/ingestion/pipeline.py) | `ingest_file`. Client exposes this. |
| [management.py](file:///c:/Environment/RAG/ultimate_rag/src/database/management.py) | `delete_document`, `get_stats`. Client exposes these. |
| [vector_search.py](file:///c:/Environment/RAG/ultimate_rag/src/agent/tools/vector_search.py) | Logic for direct search functionality. |

### New Files to Create

| File | Purpose |
|------|---------|
| `src/client.py` | The RAGClient implementation. |
| `tests/test_client.py` | Integration tests for the client. |

### Patterns to Follow

**Async Context Manager:**
```python
async with RAGClient() as client:
    await client.query("...")
```

**Facade Pattern:**
Hide complex imports. Users only import `RAGClient`.

---

## IMPLEMENTATION PLAN

### Phase 1: Core Client Structure

Create `src/client.py` with `RAGClient` class.

**Key Methods:**
- `__init__(session_id: str = "default", config: Settings = None)`
- `__aenter__ / __aexit__`: Resource management.
- `ingest(path)`: Single file.
- `ingest_batch(paths)`: Multiple files (concurrent).
- `query(text, session_id)`: Agent interaction.
- `search(query, limit)`: Direct vector search (bypass agent).
- `management`: Expose all DB management functions.

### Phase 2: Direct Search Implementation

Refactor `vector_search.py` logic slightly if needed, or just reuse the tool function logic within `client.search()` to allow "search without agent".

### Phase 3: Testing

Comprehensive integration tests using `pytest-asyncio`.

---

## STEP-BY-STEP TASKS

### CREATE `src/client.py` - Imports & Class Def

```python
"""RAG Client facade."""
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, List

from src.config import get_settings, Settings
from src.agent import create_agent, RAGAgent
from src.ingestion import ingest_file
from src.database import (
    get_connection,
    close_pool,
    delete_document,
    document_exists,
    update_document,
    get_stats,
    cleanup_duplicates,
    cleanup_orphan_chunks,
    truncate_all_tables,
)
# We might need to import vector_search logic directly or use the tool
from src.agent.tools.vector_search import vector_search_tool 

logger = logging.getLogger(__name__)

class RAGClient:
    """High-level client for the RAG system."""

    def __init__(self, session_id: str = "default", settings: Optional[Settings] = None):
        self.session_id = session_id
        self.settings = settings or get_settings()
        self._agent: Optional[RAGAgent] = None
        self._current_agent_sid: Optional[str] = None
```

- **IMPLEMENT**: Docstrings and initial setup.
- **VALIDATE**: `ruff check src/client.py`

---

### IMPLEMENT Context Manager

```python
    async def __aenter__(self) -> "RAGClient":
        """Initialize resources/pool."""
        # Ensure pool is ready (get_connection does this lazily, but good to warm up)
        async with get_connection():
            pass 
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources."""
        await close_pool()
```

- **IMPLEMENT**: Standard async context protocol.

---

### IMPLEMENT Ingestion Methods

```python
    async def ingest(self, file_path: str | Path) -> str:
        """Ingest a single document."""
        return await ingest_file(file_path)

    async def ingest_batch(self, file_paths: List[str | Path]) -> List[str]:
        """Ingest multiple documents concurrently."""
        tasks = [self.ingest(p) for p in file_paths]
        return await asyncio.gather(*tasks)
```

- **IMPLEMENT**: `ingest` and `ingest_batch`.
- **PATTERN**: `asyncio.gather` for concurrency.

---

### IMPLEMENT Agent Methods

```python
    async def _ensure_agent(self, session_id: str) -> RAGAgent:
        if not self._agent or self._current_agent_sid != session_id:
            self._agent = await create_agent(session_id)
            self._current_agent_sid = session_id
        return self._agent

    async def query(self, input_text: str, session_id: str = None) -> str:
        """Query the RAG agent."""
        sid = session_id or self.session_id
        agent = await self._ensure_agent(sid)
        response = await agent.invoke(input_text)
        return response.content
    
    async def search(self, query: str, limit: int = 5) -> str:
        """Direct vector search (bypassing agent personality)."""
        # Reuses the tool logic directly
        return await vector_search_tool.ainvoke({"query": query, "limit": limit})
```

- **IMPLEMENT**: `query` and `search`.
- **NOTE**: Using `vector_search_tool.ainvoke` is a clever reuse of existing logic.

---

### IMPLEMENT Management Methods

```python
    async def delete_document(self, file_id: str) -> bool:
        return await delete_document(file_id)

    async def document_exists(self, file_id: str) -> bool:
        return await document_exists(file_id)

    async def get_stats(self) -> dict[str, Any]:
        return await get_stats()

    async def cleanup(self) -> dict[str, int]:
        """Run all cleanup tasks."""
        dups = await cleanup_duplicates()
        orphans = await cleanup_orphan_chunks()
        return {"duplicates_removed": dups, "orphans_removed": orphans}
```

- **IMPLEMENT**: Wrappers for management functions.

---

### UPDATE `src/__init__.py`

```python
from .client import RAGClient
__all__ = ["RAGClient", "get_settings", ...] 
```

- **IMPLEMENT**: Export Client.

---

### CREATE `tests/test_client.py`

```python
import pytest
from src.client import RAGClient

@pytest.mark.asyncio
async def test_client_context_manager():
    async with RAGClient() as client:
        stats = await client.get_stats()
        assert isinstance(stats, dict)

@pytest.mark.asyncio
async def test_client_ingest_and_query():
    # Mocking would be ideal here, but for now we test structure
    pass 
```

- **IMPLEMENT**: Basic structural tests.

---

## TESTING STRATEGY

### Unit Tests
- Test Context Manager behavior (does it close pool?)
- Test `ingest_batch` concurrency.
- Test `search` direct return.

### Integration Tests
- Full flow: `async with RAGClient() as c: await c.ingest(...) -> await c.query(...)`

---

## VALIDATION COMMANDS

### Level 1: Syntax
```bash
ruff check src/client.py
```

### Level 2: Import Check
```bash
python -c "from src import RAGClient"
```

### Level 3: Unit Tests
```bash
pytest tests/test_client.py -v
```

### Level 4: Live Check
```bash
python -c "import asyncio; from src.client import RAGClient; asyncio.run(RAGClient().get_stats())"
```

---

## ACCEPTANCE CRITERIA
- [ ] `RAGClient` supports `async with`
- [ ] `ingest_batch` works concurrently
- [ ] `search` returns results without agent invocation
- [ ] `cleanup` combines multiple cleanup utils
- [ ] All tests pass

---

## COMPLETION CHECKLIST
- [ ] `src/client.py` fully implemented
- [ ] `src/__init__.py` updated
- [ ] `tests/test_client.py` created and passed
