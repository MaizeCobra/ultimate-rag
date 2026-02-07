# Feature: Phase 3.5 - Database Management Module

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to existing async patterns in `connection.py` and `pipeline.py`.

## Feature Description

Database management operations for the Ultimate RAG system, matching n8n workflow patterns. Provides CRUD operations for documents, cleanup utilities for duplicates/orphans, and admin functions for database maintenance.

## User Story

As a RAG system administrator
I want to manage documents in my knowledge base (delete, update, cleanup duplicates)
So that I can keep my knowledge base clean and up-to-date

## Problem Statement

Currently the system can only INSERT documents. There's no way to:
- Delete documents (removes from all 3 tables)
- Update documents (delete + re-ingest)
- Clean up duplicate entries from multiple ingestion runs
- Get database statistics for monitoring

## Solution Statement

Create `src/database/management.py` with async functions that mirror n8n workflow SQL patterns, providing complete lifecycle management for documents.

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: Medium
**Primary Systems Affected**: database module
**Dependencies**: asyncpg (already installed)

---

## CONTEXT REFERENCES

### Relevant Codebase Files - READ BEFORE IMPLEMENTING!

| File | Lines | Why |
|------|-------|-----|
| [connection.py](file:///c:/Environment/RAG/ultimate_rag/src/database/connection.py) | 33-38 | Pattern: `async with get_connection() as conn` |
| [schema.sql](file:///c:/Environment/RAG/ultimate_rag/src/database/schema.sql) | 1-38 | Table structures and constraints |
| [pipeline.py](file:///c:/Environment/RAG/ultimate_rag/src/ingestion/pipeline.py) | 109-129 | Pattern: UPSERT with ON CONFLICT |
| [memory.py](file:///c:/Environment/RAG/ultimate_rag/src/agent/memory.py) | 83-86 | Pattern: DELETE with session_id |
| [test_database.py](file:///c:/Environment/RAG/ultimate_rag/tests/test_database.py) | 60-91 | Test patterns with cleanup |

### n8n Workflow SQL Patterns (Ultimate RAG.json)

**Delete from documents_pg** (line 824):
```sql
DELETE FROM documents_pg WHERE metadata->>'file_id' = $1
```

**Delete from document_rows** (line 847):
```sql
DELETE FROM document_rows WHERE dataset_id = $1
```

**Delete from document_metadata** (line 1017):
```sql
DELETE FROM document_metadata WHERE id = $1
```

### New Files to Create

| File | Purpose |
|------|---------|
| `src/database/management.py` | All DB management functions |
| `tests/test_management.py` | Unit tests for management |

### Patterns to Follow

**Async Context Manager:**
```python
async with get_connection() as conn:
    result = await conn.execute(sql, param)
```

**SQL Parameterization:**
```python
await conn.execute("DELETE FROM table WHERE id = $1", file_id)
```

**Return Types:**
- `bool` for success/failure (delete_document)
- `int` for count of affected rows (cleanup functions)
- `dict` for stats and complex returns

---

## IMPLEMENTATION PLAN

### Phase 1: Core CRUD Operations

Create `management.py` with:
- `delete_document(file_id)` - cascade delete from all tables
- `document_exists(file_id)` - check existence
- `update_document(file_id, file_path)` - delete + re-ingest

### Phase 2: Cleanup Utilities

Add:
- `get_duplicate_files()` - group by filename, show duplicates
- `cleanup_duplicates()` - keep first by created_at
- `cleanup_orphan_chunks()` - chunks without metadata

### Phase 3: Admin Functions

Add:
- `get_stats()` - table counts and sizes
- `truncate_all_tables()` - reset with confirmation

### Phase 4: Testing

Create comprehensive tests matching project patterns.

---

## STEP-BY-STEP TASKS

### CREATE `src/database/management.py`

```python
"""Database management operations."""
import json
import logging
from typing import Any

from .connection import get_connection

logger = logging.getLogger(__name__)
```

- **IMPLEMENT**: Module docstring and imports
- **PATTERN**: Follow `connection.py` style
- **VALIDATE**: `ruff check src/database/management.py`

---

### ADD `document_exists(file_id: str) -> bool`

```python
async def document_exists(file_id: str) -> bool:
    """Check if a document exists in the database."""
    async with get_connection() as conn:
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM document_metadata WHERE id = $1)",
            file_id
        )
        return result
```

- **IMPLEMENT**: Simple existence check
- **PATTERN**: `fetchval` for single value
- **VALIDATE**: Test with known file_id

---

### ADD `delete_document(file_id: str) -> bool`

```python
async def delete_document(file_id: str) -> bool:
    """Delete document from all tables.
    
    Order: documents_pg -> document_rows -> document_metadata
    (metadata has cascade, but explicit is safer)
    """
    async with get_connection() as conn:
        # 1. Delete from vector store
        await conn.execute(
            "DELETE FROM documents_pg WHERE metadata->>'file_id' = $1",
            file_id
        )
        
        # 2. Delete from tabular rows
        await conn.execute(
            "DELETE FROM document_rows WHERE dataset_id = $1",
            file_id
        )
        
        # 3. Delete metadata (will cascade if missed above)
        result = await conn.execute(
            "DELETE FROM document_metadata WHERE id = $1",
            file_id
        )
        
        deleted = "DELETE 1" in result
        logger.info(f"Deleted document {file_id}: {deleted}")
        return deleted
```

- **IMPLEMENT**: Cascade delete matching n8n pattern
- **PATTERN**: n8n DELETE queries (lines 824, 847, 1017)
- **GOTCHA**: Must delete in correct order (vectors → rows → metadata)
- **VALIDATE**: `python -c "import asyncio; from src.database.management import delete_document; print(asyncio.run(delete_document('test')))"`

---

### ADD `update_document(file_id: str, file_path: str) -> str`

```python
async def update_document(file_id: str, file_path: str) -> str:
    """Update document by deleting and re-ingesting."""
    from src.ingestion import ingest_file
    
    # Delete existing
    await delete_document(file_id)
    
    # Re-ingest (generates new file_id)
    new_file_id = await ingest_file(file_path)
    logger.info(f"Updated {file_id} -> {new_file_id}")
    return new_file_id
```

- **IMPLEMENT**: Delete + re-ingest flow
- **IMPORTS**: Lazy import to avoid circular
- **VALIDATE**: Test with existing file

---

### ADD `get_duplicate_files() -> list[dict]`

```python
async def get_duplicate_files() -> list[dict]:
    """Find duplicate files by filename."""
    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT filename, COUNT(*) as count, 
                   array_agg(id ORDER BY created_at) as file_ids
            FROM document_metadata
            GROUP BY filename
            HAVING COUNT(*) > 1
        """)
        return [dict(r) for r in rows]
```

- **IMPLEMENT**: GROUP BY with HAVING
- **VALIDATE**: `python tests/check_docs.py` should show duplicates

---

### ADD `cleanup_duplicates() -> int`

```python
async def cleanup_duplicates() -> int:
    """Remove duplicate files, keeping first by created_at."""
    duplicates = await get_duplicate_files()
    deleted = 0
    
    for dup in duplicates:
        # Keep first (oldest), delete rest
        ids_to_delete = dup["file_ids"][1:]  # Skip first
        for file_id in ids_to_delete:
            if await delete_document(file_id):
                deleted += 1
    
    logger.info(f"Cleaned up {deleted} duplicate documents")
    return deleted
```

- **IMPLEMENT**: Keep oldest, delete newer duplicates
- **VALIDATE**: Before/after document counts

---

### ADD `cleanup_orphan_chunks() -> int`

```python
async def cleanup_orphan_chunks() -> int:
    """Remove chunks without corresponding metadata."""
    async with get_connection() as conn:
        result = await conn.execute("""
            DELETE FROM documents_pg
            WHERE metadata->>'file_id' NOT IN (
                SELECT id FROM document_metadata
            )
        """)
        # Parse "DELETE N" result
        count = int(result.split()[-1]) if "DELETE" in result else 0
        logger.info(f"Cleaned up {count} orphan chunks")
        return count
```

- **IMPLEMENT**: Delete orphans using NOT IN subquery
- **VALIDATE**: Artificially create orphan, then cleanup

---

### ADD `get_stats() -> dict`

```python
async def get_stats() -> dict:
    """Get database statistics."""
    async with get_connection() as conn:
        stats = {}
        
        for table in ["documents_pg", "document_metadata", "document_rows", "chat_memory"]:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            stats[table] = count
        
        # Add duplicate count
        duplicates = await get_duplicate_files()
        stats["duplicate_files"] = len(duplicates)
        
        return stats
```

- **IMPLEMENT**: Table counts plus duplicate info
- **VALIDATE**: Output matches expected structure

---

### ADD `truncate_all_tables() -> None`

```python
async def truncate_all_tables() -> None:
    """Reset all tables. USE WITH CAUTION."""
    async with get_connection() as conn:
        await conn.execute("TRUNCATE documents_pg CASCADE")
        await conn.execute("TRUNCATE document_rows CASCADE")
        await conn.execute("TRUNCATE document_metadata CASCADE")
        await conn.execute("TRUNCATE chat_memory CASCADE")
    
    logger.warning("All tables truncated!")
```

- **IMPLEMENT**: TRUNCATE with CASCADE
- **GOTCHA**: This is destructive, no undo
- **VALIDATE**: Stats should show 0 after truncate

---

### UPDATE `src/database/__init__.py`

Add exports for new functions:

```python
from .management import (
    delete_document,
    document_exists,
    update_document,
    get_duplicate_files,
    cleanup_duplicates,
    cleanup_orphan_chunks,
    get_stats,
    truncate_all_tables,
)
```

- **VALIDATE**: `from src.database import delete_document` works

---

### CREATE `tests/test_management.py`

```python
"""Tests for database management functions."""
import pytest
from src.database import get_connection
from src.database.management import (
    delete_document,
    document_exists,
    get_stats,
    get_duplicate_files,
)


@pytest.mark.asyncio
async def test_document_exists_false():
    """Non-existent document returns False."""
    exists = await document_exists("nonexistent-id")
    assert exists is False


@pytest.mark.asyncio
async def test_delete_nonexistent_document():
    """Deleting non-existent returns False."""
    result = await delete_document("nonexistent-id")
    assert result is False


@pytest.mark.asyncio
async def test_get_stats_structure():
    """Stats returns expected keys."""
    stats = await get_stats()
    assert "documents_pg" in stats
    assert "document_metadata" in stats
    assert "duplicate_files" in stats
```

- **PATTERN**: Follow `test_database.py` style
- **VALIDATE**: `pytest tests/test_management.py -v`

---

## TESTING STRATEGY

### Unit Tests

- Test each function with mock data
- Test edge cases (empty results, non-existent IDs)
- Test return types match signatures

### Integration Tests

- Create test document → verify exists → delete → verify gone
- Create duplicates → cleanup → verify single remains

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style
```bash
ruff check src/database/management.py
ruff check tests/test_management.py
```

### Level 2: Unit Tests
```bash
pytest tests/test_management.py -v
```

### Level 3: Live Validation
```bash
python -c "
import asyncio
from src.database.management import get_stats
stats = asyncio.run(get_stats())
print(stats)
"
```

### Level 4: Cleanup Test
```bash
python -c "
import asyncio
from src.database.management import cleanup_duplicates, get_stats
before = asyncio.run(get_stats())
deleted = asyncio.run(cleanup_duplicates())
after = asyncio.run(get_stats())
print(f'Before: {before}')
print(f'Deleted: {deleted}')
print(f'After: {after}')
"
```

---

## ACCEPTANCE CRITERIA

- [ ] `delete_document` removes from all 3 tables
- [ ] `document_exists` correctly identifies presence
- [ ] `get_duplicate_files` finds duplicates
- [ ] `cleanup_duplicates` reduces to unique files
- [ ] `get_stats` returns accurate counts
- [ ] All functions are async
- [ ] All functions use `get_connection()` pattern
- [ ] All validation commands pass
- [ ] Unit tests pass

---

## COMPLETION CHECKLIST

- [ ] `management.py` created with all functions
- [ ] `__init__.py` exports updated
- [ ] `test_management.py` passes
- [ ] Ruff linting passes
- [ ] Live cleanup_duplicates works
- [ ] Stats accurate before/after

---

## NOTES

**Order of deletion matters:** Due to FK constraints, delete child tables first (documents_pg, document_rows) before parent (document_metadata).

**Circular import:** `update_document` must lazy-import `ingest_file` to avoid circular dependency.

**Confidence Score:** 9/10 - Well-defined patterns from existing codebase and n8n workflow.
