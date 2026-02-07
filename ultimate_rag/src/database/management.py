"""Database management operations.

Provides CRUD operations for documents, cleanup utilities for
duplicates/orphans, and admin functions for database maintenance.
Matches n8n Ultimate RAG workflow patterns.
"""
import logging
from typing import Any

from .connection import get_connection


logger = logging.getLogger(__name__)


async def document_exists(file_id: str) -> bool:
    """Check if a document exists in the database.
    
    Args:
        file_id: Document ID to check
        
    Returns:
        True if document exists, False otherwise
    """
    async with get_connection() as conn:
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM document_metadata WHERE id = $1)",
            file_id
        )
        return result


async def delete_document(file_id: str) -> bool:
    """Delete document from all tables.
    
    Deletes in order: documents_pg -> document_rows -> document_metadata
    (child tables first, then parent)
    
    Args:
        file_id: Document ID to delete
        
    Returns:
        True if document was deleted, False if not found
    """
    async with get_connection() as conn:
        # 1. Delete from vector store (chunks)
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
        if deleted:
            logger.info(f"Deleted document {file_id}")
        return deleted


async def update_document(file_id: str, file_path: str) -> str:
    """Update document by deleting and re-ingesting.
    
    Args:
        file_id: Existing document ID to replace
        file_path: Path to new file to ingest
        
    Returns:
        New document ID after re-ingestion
    """
    # Lazy import to avoid circular dependency
    from src.ingestion import ingest_file
    
    # Delete existing
    await delete_document(file_id)
    
    # Re-ingest (generates new file_id)
    new_file_id = await ingest_file(file_path)
    logger.info(f"Updated {file_id} -> {new_file_id}")
    return new_file_id


async def get_duplicate_files() -> list[dict[str, Any]]:
    """Find duplicate files by filename.
    
    Returns:
        List of dicts with filename, count, and file_ids
    """
    async with get_connection() as conn:
        rows = await conn.fetch("""
            SELECT filename, COUNT(*) as count, 
                   array_agg(id ORDER BY created_at) as file_ids
            FROM document_metadata
            GROUP BY filename
            HAVING COUNT(*) > 1
        """)
        return [dict(r) for r in rows]


async def cleanup_duplicates() -> int:
    """Remove duplicate files, keeping first by created_at.
    
    Returns:
        Number of duplicate documents deleted
    """
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


async def cleanup_orphan_chunks() -> int:
    """Remove chunks without corresponding metadata.
    
    Returns:
        Number of orphan chunks deleted
    """
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


async def get_stats() -> dict[str, Any]:
    """Get database statistics.
    
    Returns:
        Dict with table counts and duplicate file count
    """
    async with get_connection() as conn:
        stats: dict[str, Any] = {}
        
        for table in ["documents_pg", "document_metadata", "document_rows", "chat_memory"]:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            stats[table] = count
        
        # Add duplicate count
        duplicates = await get_duplicate_files()
        stats["duplicate_files"] = len(duplicates)
        
        return stats


async def truncate_all_tables() -> None:
    """Reset all tables. USE WITH CAUTION.
    
    This permanently deletes ALL data from all tables.
    Cannot be undone!
    """
    async with get_connection() as conn:
        await conn.execute("TRUNCATE documents_pg CASCADE")
        await conn.execute("TRUNCATE document_rows CASCADE")
        await conn.execute("TRUNCATE document_metadata CASCADE")
        await conn.execute("TRUNCATE chat_memory CASCADE")
    
    logger.warning("All tables truncated!")
