"""Tests for database management functions."""
import pytest
from src.database import get_connection, close_pool
from src.database.management import (
    delete_document,
    document_exists,
    get_stats,
    get_duplicate_files,
    cleanup_orphan_chunks,
)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after tests."""
    yield
    await close_pool()


@pytest.mark.asyncio
async def test_document_exists_false():
    """Non-existent document returns False."""
    exists = await document_exists("nonexistent-id-12345")
    assert exists is False


@pytest.mark.asyncio
async def test_delete_nonexistent_document():
    """Deleting non-existent returns False."""
    result = await delete_document("nonexistent-id-12345")
    assert result is False


@pytest.mark.asyncio
async def test_get_stats_structure():
    """Stats returns expected keys."""
    stats = await get_stats()
    assert "documents_pg" in stats
    assert "document_metadata" in stats
    assert "document_rows" in stats
    assert "chat_memory" in stats
    assert "duplicate_files" in stats
    # All counts should be integers
    assert isinstance(stats["documents_pg"], int)
    assert isinstance(stats["duplicate_files"], int)


@pytest.mark.asyncio
async def test_get_stats_returns_counts():
    """Stats values should be non-negative."""
    stats = await get_stats()
    for key, value in stats.items():
        assert value >= 0, f"{key} should be non-negative"


@pytest.mark.asyncio
async def test_get_duplicate_files_returns_list():
    """get_duplicate_files returns a list."""
    duplicates = await get_duplicate_files()
    assert isinstance(duplicates, list)


@pytest.mark.asyncio
async def test_cleanup_orphan_chunks_returns_int():
    """cleanup_orphan_chunks returns count."""
    count = await cleanup_orphan_chunks()
    assert isinstance(count, int)
    assert count >= 0


@pytest.mark.asyncio
async def test_document_exists_with_real_doc():
    """Test exists with a real document if any exist."""
    stats = await get_stats()
    if stats["document_metadata"] > 0:
        # Get a real document ID
        async with get_connection() as conn:
            row = await conn.fetchrow("SELECT id FROM document_metadata LIMIT 1")
            if row:
                exists = await document_exists(row["id"])
                assert exists is True
