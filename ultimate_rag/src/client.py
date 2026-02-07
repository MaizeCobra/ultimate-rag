"""RAG Client facade.

This module provides a high-level, production-ready RAGClient that transforms
the system into a reusable library. It wraps ingestion, querying, and database
management into a single class with context manager support.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.agent.tools.vector_search import vector_search_tool

logger = logging.getLogger(__name__)


class RAGClient:
    """High-level client for the Ultimate RAG system.

    Usage:
        async with RAGClient() as client:
            await client.ingest("doc.pdf")
            response = await client.query("What is in the doc?")
    """

    def __init__(self, session_id: str = "default", settings: Optional[Settings] = None):
        """Initialize the RAG client.

        Args:
            session_id: Default session ID for agent interactions.
            settings: Optional configuration overrides.
        """
        self.session_id = session_id
        self.settings = settings or get_settings()
        self._agent: Optional[RAGAgent] = None
        self._current_agent_sid: Optional[str] = None

    async def __aenter__(self) -> "RAGClient":
        """Initialize resources and connection pool."""
        # Warm up the connection pool
        async with get_connection():
            pass
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources and close connection pool."""
        await close_pool()

    async def ingest(self, file_path: str | Path) -> str:
        """Ingest a single document.

        Args:
            file_path: Path to the file.

        Returns:
            The document ID.
        """
        return await ingest_file(file_path)

    async def ingest_batch(self, file_paths: List[str | Path]) -> List[str]:
        """Ingest multiple documents concurrently.

        Args:
            file_paths: List of file paths.

        Returns:
            List of document IDs.
        """
        tasks = [self.ingest(p) for p in file_paths]
        return await asyncio.gather(*tasks)

    async def _ensure_agent(self, session_id: str) -> RAGAgent:
        """Get or create the agent for the specified session."""
        if not self._agent or self._current_agent_sid != session_id:
            self._agent = await create_agent(session_id)
            self._current_agent_sid = session_id
        return self._agent

    async def query(self, input_text: str, session_id: str = None) -> str:
        """Query the RAG agent.

        Args:
            input_text: User question or command.
            session_id: Optional session ID override.

        Returns:
            The agent's text response.
        """
        sid = session_id or self.session_id
        agent = await self._ensure_agent(sid)
        response = await agent.invoke(input_text)
        return response.content

    async def search(self, query: str, limit: int = 5) -> str:
        """Direct vector search (bypassing agent personality).

        Useful for raw retrieval without conversational wrapper.

        Args:
            query: The search query string.
            limit: Maximum number of results.

        Returns:
            Formatted search results string.
        """
        # Reuses the existing vector search tool logic directly
        return await vector_search_tool.ainvoke({"query": query, "limit": limit})

    async def delete_document(self, file_id: str) -> bool:
        """Delete a document by ID."""
        return await delete_document(file_id)
    
    async def update_document(self, file_id: str, file_path: str) -> str:
        """Update a document by re-ingesting it."""
        return await update_document(file_id, file_path)

    async def document_exists(self, file_id: str) -> bool:
        """Check if a document exists."""
        return await document_exists(file_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return await get_stats()

    async def cleanup(self) -> Dict[str, int]:
        """Run all cleanup tasks (duplicates and orphans).

        Returns:
            Dictionary with counts of removed items.
        """
        dups = await cleanup_duplicates()
        orphans = await cleanup_orphan_chunks()
        return {"duplicates_removed": dups, "orphans_removed": orphans}

    async def truncate_all_tables(self) -> None:
        """Reset the entire database. DESTRUCTIVE."""
        await truncate_all_tables()
