"""Integration tests for RAGClient."""
import pytest
from unittest.mock import AsyncMock, patch

from src.client import RAGClient

@pytest.mark.asyncio
async def test_client_context_manager():
    """Test that context manager opens and closes resources."""
    with patch("src.client.get_connection") as mock_get_conn, \
         patch("src.client.close_pool") as mock_close_pool:
        
        mock_conn = AsyncMock()
        mock_get_conn.return_value.__aenter__.return_value = mock_conn

        async with RAGClient() as client:
            assert isinstance(client, RAGClient)
            # get_connection should be called to warm up pool
            mock_get_conn.assert_called()
        
        # close_pool should be called on exit
        mock_close_pool.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_ingest_batch():
    """Test batch ingestion concurrency."""
    with patch("src.client.ingest_file", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.side_effect = ["id1", "id2"]
        
        client = RAGClient()
        file_paths = ["doc1.pdf", "doc2.pdf"]
        
        results = await client.ingest_batch(file_paths)
        
        assert results == ["id1", "id2"]
        assert mock_ingest.call_count == 2
        # Verify calls were made
        mock_ingest.assert_any_call("doc1.pdf")
        mock_ingest.assert_any_call("doc2.pdf")

@pytest.mark.asyncio
async def test_client_query_agent_management():
    """Test agent creation and reuse."""
    with patch("src.client.create_agent", new_callable=AsyncMock) as mock_create_agent:
        mock_agent = AsyncMock()
        mock_agent.invoke.return_value.content = "Response"
        mock_create_agent.return_value = mock_agent
        
        client = RAGClient(session_id="session-1")
        
        # First query - creates agent
        response1 = await client.query("Hello")
        assert response1 == "Response"
        mock_create_agent.assert_called_with("session-1")
        
        # Second query, same session - reuses agent
        mock_create_agent.reset_mock()
        await client.query("Again")
        mock_create_agent.assert_not_called()
        
        # Third query, DIFFERENT session - creates new agent
        await client.query("New check", session_id="session-2")
        mock_create_agent.assert_called_with("session-2")

@pytest.mark.asyncio
async def test_client_cleanup():
    """Test cleanup implementation."""
    with patch("src.client.cleanup_duplicates", new_callable=AsyncMock) as mock_dups, \
         patch("src.client.cleanup_orphan_chunks", new_callable=AsyncMock) as mock_orphans:
        
        mock_dups.return_value = 5
        mock_orphans.return_value = 10
        
        client = RAGClient()
        stats = await client.cleanup()
        
        assert stats["duplicates_removed"] == 5
        assert stats["orphans_removed"] == 10
        mock_dups.assert_awaited_once()
        mock_orphans.assert_awaited_once()
