"""Database connection and schema tests."""
import json
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
            VALUES ($1, $2::jsonb)
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
        
        # asyncpg returns JSONB as a string, need to parse it
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert metadata["file_id"] == "test123"
        
        # Cleanup
        await conn.execute("DELETE FROM documents_pg WHERE content = $1", "Test content")
