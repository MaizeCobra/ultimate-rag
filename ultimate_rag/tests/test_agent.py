"""Tests for the RAG agent and tools."""
import pytest
import json
from datetime import datetime

from src.agent.memory import PostgresChatMemory, ChatMessage
from src.agent.prompts import RAG_SYSTEM_PROMPT


class TestChatMemory:
    """Tests for PostgresChatMemory."""
    
    def test_init_sets_session_id(self):
        """Test initialization sets session ID."""
        memory = PostgresChatMemory(session_id="test-session")
        assert memory.session_id == "test-session"
        assert memory.max_messages == 10
    
    def test_init_custom_max_messages(self):
        """Test custom max_messages."""
        memory = PostgresChatMemory(session_id="test", max_messages=20)
        assert memory.max_messages == 20
    
    @pytest.mark.asyncio
    async def test_add_message(self, mock_db_connection):
        """Test adding a message to memory."""
        memory = PostgresChatMemory(session_id="test-session")
        await memory.add_message("human", "Hello")
        # Verify DB execute was called
        mock_db_connection.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_messages_empty(self, mock_db_connection):
        """Test getting messages when empty."""
        mock_db_connection.fetch.return_value = []
        memory = PostgresChatMemory(session_id="test-session")
        messages = await memory.get_messages()
        assert messages == []
    
    @pytest.mark.asyncio
    async def test_get_messages_chronological(self, mock_db_connection):
        """Test messages are returned in chronological order."""
        mock_db_connection.fetch.return_value = [
            {"role": "ai", "content": "Hi!", "created_at": datetime(2024, 1, 1, 12, 1)},
            {"role": "human", "content": "Hello", "created_at": datetime(2024, 1, 1, 12, 0)},
        ]
        memory = PostgresChatMemory(session_id="test-session")
        messages = await memory.get_messages()
        # Should be reversed (oldest first)
        assert messages[0].role == "human"
        assert messages[1].role == "ai"
    
    @pytest.mark.asyncio
    async def test_clear(self, mock_db_connection):
        """Test clearing messages."""
        memory = PostgresChatMemory(session_id="test-session")
        await memory.clear()
        mock_db_connection.execute.assert_called_once()


class TestChatMessage:
    """Tests for ChatMessage dataclass."""
    
    def test_create_message(self):
        """Test creating a chat message."""
        msg = ChatMessage(role="human", content="Hello")
        assert msg.role == "human"
        assert msg.content == "Hello"
        assert msg.created_at is None
    
    def test_create_message_with_timestamp(self):
        """Test message with timestamp."""
        now = datetime.now()
        msg = ChatMessage(role="ai", content="Hi!", created_at=now)
        assert msg.created_at == now


class TestSystemPrompt:
    """Tests for system prompts."""
    
    def test_system_prompt_not_empty(self):
        """Ensure system prompt is defined."""
        assert len(RAG_SYSTEM_PROMPT) > 100
    
    def test_system_prompt_mentions_rag(self):
        """System prompt should mention RAG."""
        assert "RAG" in RAG_SYSTEM_PROMPT
    
    def test_system_prompt_mentions_sql(self):
        """System prompt should mention SQL queries."""
        assert "SQL" in RAG_SYSTEM_PROMPT
    
    def test_system_prompt_mentions_tables(self):
        """System prompt should mention specific tables."""
        assert "document_metadata" in RAG_SYSTEM_PROMPT
        assert "document_rows" in RAG_SYSTEM_PROMPT


class TestSqlQueryToolSecurity:
    """Tests for SQL query tool security."""
    
    @pytest.mark.asyncio
    async def test_blocks_drop_statement(self):
        """Test that DROP statements are blocked."""
        from src.agent.tools.sql_query import sql_query_tool
        
        result = await sql_query_tool.ainvoke({
            "sql_query": "DROP TABLE document_rows",
            "dataset_id": "test"
        })
        parsed = json.loads(result)
        
        assert "error" in parsed
        assert "DROP" in parsed["error"]
    
    @pytest.mark.asyncio
    async def test_blocks_delete_statement(self):
        """Test that DELETE statements are blocked."""
        from src.agent.tools.sql_query import sql_query_tool
        
        result = await sql_query_tool.ainvoke({
            "sql_query": "DELETE FROM document_rows",
            "dataset_id": "test"
        })
        parsed = json.loads(result)
        
        assert "error" in parsed
        assert "DELETE" in parsed["error"]
    
    @pytest.mark.asyncio
    async def test_requires_document_rows_table(self):
        """Test that query must reference document_rows."""
        from src.agent.tools.sql_query import sql_query_tool
        
        result = await sql_query_tool.ainvoke({
            "sql_query": "SELECT * FROM users",
            "dataset_id": "test"
        })
        parsed = json.loads(result)
        
        assert "error" in parsed
        assert "document_rows" in parsed["error"].lower()
