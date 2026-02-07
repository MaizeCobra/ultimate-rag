# Feature: Phase 3 - Retrieval System & AI Agent

The following plan should be complete, but validate documentation and codebase patterns before implementing.

Pay special attention to naming of existing utils, types, and models. Import from the right files.

---

## Feature Description

Implement the retrieval system and AI agent for the Ultimate RAG system. This includes:
1. **Vector Search Tool** - pgvector cosine similarity search with top_k=25
2. **Re-ranker Tool** (Optional) - Google Discovery Engine semantic re-ranking to top 5
3. **List Documents Tool** - SQL query to browse available documents
4. **Get File Contents Tool** - Retrieve full document text by file_id
5. **SQL Query Tool** - Dynamic SQL queries against tabular data (document_rows)
6. **Chat Memory** - PostgreSQL-based conversation history
7. **RAG AI Agent** - LangChain agent orchestrating all tools with Gemini 2.5 Pro

## User Story

As a **user interacting with the RAG system**
I want to **ask questions about my documents in natural language**
So that **I can find information quickly without manually searching**

## Problem Statement

Phase 2 implemented document ingestion and storage. Users need an intelligent interface to:
- Search documents semantically via vector similarity
- Browse and explore available documents
- Query tabular data with SQL
- Have coherent multi-turn conversations with memory

## Solution Statement

Build a LangChain-based AI agent with 5+ tools matching the n8n workflow exactly:
1. Use `@tool` decorator for custom tools
2. Use `create_agent` for ReAct agent with Gemini 2.5 Pro
3. Use PostgreSQL for chat memory (existing `chat_memory` table)
4. Maintain feature parity with n8n "Ultimate RAG" workflow

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: High
**Primary Systems Affected**: `src/agent/`, `src/api/`
**Dependencies**: google-generativeai, langchain, langchain-google-genai, asyncpg

---

## CONTEXT REFERENCES

### Relevant Codebase Files (READ BEFORE IMPLEMENTING!)

- [embedder.py](file:///c:/Environment/RAG/ultimate_rag/src/ingestion/embedder.py) - **Why**: Reuse embedding logic for query embedding
- [connection.py](file:///c:/Environment/RAG/ultimate_rag/src/database/connection.py) - **Why**: Async database connection pattern (asyncpg)
- [schema.sql](file:///c:/Environment/RAG/ultimate_rag/src/database/schema.sql) - **Why**: Table schemas for `documents_pg`, `document_metadata`, `document_rows`, `chat_memory`
- [config.py](file:///c:/Environment/RAG/ultimate_rag/src/config.py) - **Why**: Settings access pattern, model names
- [pipeline.py](file:///c:/Environment/RAG/ultimate_rag/src/ingestion/pipeline.py) - **Why**: Example of using get_connection() and JSON handling

### N8N Workflow Reference (CRITICAL!)

- [Ultimate RAG.json](file:///c:/Environment/RAG/Ultimate%20RAG.json) - **Main workflow**:
  - Lines 316-330: RAG AI Agent with system prompt
  - Lines 777-798: pgvector store config (top_k=25)
  - Lines 166-182: Postgres Chat Memory node
  - Lines 1131-1149: Gemini Chat Model (gemini-2.5-pro)

- [vector_store_+_re-ranker.json](file:///c:/Environment/RAG/vector_store_+_re-ranker.json) - **Re-ranker sub-workflow**:
  - Lines 24-44: pgvector load mode (top_k=25)
  - Lines 86-111: Google Discovery Engine API call
  - Line 94: `semantic-ranker-512@latest` model, topN=5

### New Files to Create

```
src/agent/
├── __init__.py          # Exports
├── agent.py             # Main RAG agent using create_agent
├── memory.py            # PostgreSQL chat memory
├── prompts.py           # System prompts
└── tools/
    ├── __init__.py      # Tool exports
    ├── vector_search.py # RAG vector search tool
    ├── reranker.py      # Google Discovery Engine re-ranker
    ├── list_docs.py     # List available documents
    ├── get_content.py   # Get full document content
    └── sql_query.py     # Query tabular data
```

### Relevant Documentation (READ BEFORE IMPLEMENTING!)

- [LangChain @tool decorator](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
  - How to define custom tools with docstrings
  - Args pattern for tool parameters

- [LangChain create_agent](https://docs.langchain.com/oss/python/langchain/sql-agent)
  - Creating ReAct agent with tools
  - System prompt configuration

- [LangChain PostgreSQL Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
  - AsyncPostgresStore for async operations
  - Chat message storage patterns

- [Google Discovery Engine Rank API](https://cloud.google.com/generative-ai-app-builder/docs/ranking)
  - POST to `/rankingConfigs/default_ranking_config:rank`
  - Request format: `{ model, query, records, topN }`

### Patterns to Follow

**Tool Definition Pattern** (from LangChain docs):
```python
from langchain.tools import tool

@tool
def my_tool(query: str) -> str:
    """Tool description for the agent.
    
    Args:
        query: The search query
    """
    return "result"
```

**Database Query Pattern** (from connection.py):
```python
from src.database import get_connection

async with get_connection() as conn:
    rows = await conn.fetch("SELECT * FROM table WHERE id = $1", id)
```

**Embedding Pattern** (from embedder.py):
```python
from src.ingestion.embedder import Embedder

embedder = Embedder()
query_embedding = await embedder.embed(query_text)
```

**Config Access Pattern** (from config.py):
```python
from src.config import get_settings

settings = get_settings()
model_name = settings.gemini_chat_model
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation - Memory & Prompts

Set up chat memory and system prompts before building tools.

**Tasks:**
1. Create PostgreSQL chat memory class
2. Create system prompts matching n8n exactly
3. Update agent `__init__.py` with exports

### Phase 2: Core Tools

Build the 5 tools that the agent will use.

**Tasks:**
1. Vector Search Tool - pgvector cosine similarity
2. List Documents Tool - SQL SELECT from document_metadata
3. Get File Contents Tool - Aggregate chunks by file_id
4. SQL Query Tool - Execute dynamic SQL on document_rows
5. (Optional) Re-ranker Tool - Google Discovery Engine

### Phase 3: AI Agent

Assemble the agent with all tools.

**Tasks:**
1. Create main agent using create_agent
2. Wire up tools, memory, and model
3. Implement async invoke method

### Phase 4: Testing & Validation

Test each component individually and integrated.

**Tasks:**
1. Unit tests for each tool
2. Integration test for agent with sample queries
3. Memory persistence test

---

## STEP-BY-STEP TASKS

Execute every task in order, top to bottom. Each task is atomic and independently testable.

---

### CREATE `src/agent/memory.py`

**IMPLEMENT**: PostgreSQL chat memory using asyncpg

```python
"""PostgreSQL-based chat memory for conversation history."""
import json
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.database import get_connection


@dataclass
class ChatMessage:
    """Single chat message."""
    role: str  # 'human' or 'ai'
    content: str
    created_at: Optional[datetime] = None


class PostgresChatMemory:
    """Manages chat history in PostgreSQL.
    
    Uses the existing chat_memory table from schema.sql.
    """
    
    def __init__(self, session_id: str, max_messages: int = 10):
        """Initialize memory for a session.
        
        Args:
            session_id: Unique session identifier
            max_messages: Maximum messages to retrieve (context window)
        """
        self.session_id = session_id
        self.max_messages = max_messages
    
    async def add_message(self, role: str, content: str) -> None:
        """Add a message to memory.
        
        Args:
            role: 'human' or 'ai'
            content: Message content
        """
        async with get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO chat_memory (session_id, role, content)
                VALUES ($1, $2, $3)
                """,
                self.session_id, role, content
            )
    
    async def get_messages(self) -> List[ChatMessage]:
        """Get recent messages for this session.
        
        Returns:
            List of ChatMessage objects, oldest first
        """
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, created_at
                FROM chat_memory
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                self.session_id, self.max_messages
            )
        
        # Reverse to get chronological order
        messages = [
            ChatMessage(role=r["role"], content=r["content"], created_at=r["created_at"])
            for r in reversed(rows)
        ]
        return messages
    
    async def clear(self) -> None:
        """Clear all messages for this session."""
        async with get_connection() as conn:
            await conn.execute(
                "DELETE FROM chat_memory WHERE session_id = $1",
                self.session_id
            )
```

- **PATTERN**: Database access from `connection.py`
- **VALIDATE**: `python -c "from src.agent.memory import PostgresChatMemory; print('OK')"`

---

### CREATE `src/agent/prompts.py`

**IMPLEMENT**: System prompts matching n8n exactly (line 320 of Ultimate RAG.json)

```python
"""System prompts for the RAG agent."""

RAG_SYSTEM_PROMPT = """You are a personal assistant who helps answer questions from a corpus of documents. The documents are either text based (Txt, docs, extracted PDFs, etc.) or tabular data (CSVs or Excel documents).

You are given tools to perform RAG in the 'documents' table, look up the documents available in your knowledge base in the 'document_metadata' table, extract all the text from a given document, and query the tabular files with SQL in the 'document_rows' table.

Always start by performing RAG unless the question requires a SQL query for tabular data (fetching a sum, finding a max, something a RAG lookup would be unreliable for). If RAG doesn't help, then look at the documents that are available to you, find a few that you think would contain the answer, and then analyze those.

Always tell the user if you didn't find the answer. Don't make something up just to please them."""

# Tool descriptions (matching n8n)
VECTOR_SEARCH_DESCRIPTION = "Use RAG to look up information in the knowledgebase."

RERANKER_DESCRIPTION = "Use RAG to look up information in the knowledgebase but with better re-ranked information"

LIST_DOCS_DESCRIPTION = "List all available documents in the knowledge base with their metadata."

GET_CONTENT_DESCRIPTION = "Get the complete text content of a specific document by its file_id."

SQL_QUERY_DESCRIPTION = "Execute a SQL query against tabular data stored in document_rows. Use JSONB operators for column access."
```

- **PATTERN**: Direct copy from n8n workflow JSON
- **VALIDATE**: `python -c "from src.agent.prompts import RAG_SYSTEM_PROMPT; print(len(RAG_SYSTEM_PROMPT))"`

---

### CREATE `src/agent/tools/__init__.py`

**IMPLEMENT**: Tool exports

```python
"""Agent tools for RAG operations."""
from .vector_search import vector_search_tool
from .list_docs import list_docs_tool
from .get_content import get_content_tool
from .sql_query import sql_query_tool

__all__ = [
    "vector_search_tool",
    "list_docs_tool",
    "get_content_tool",
    "sql_query_tool",
]
```

- **VALIDATE**: (Defer until tools created)

---

### CREATE `src/agent/tools/vector_search.py`

**IMPLEMENT**: Vector search using pgvector cosine similarity

```python
"""Vector search tool using pgvector."""
import json
from typing import List
from langchain.tools import tool

from src.database import get_connection
from src.ingestion.embedder import Embedder
from src.agent.prompts import VECTOR_SEARCH_DESCRIPTION


# Singleton embedder
_embedder = None

def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@tool
async def vector_search_tool(query: str) -> str:
    """Use RAG to look up information in the knowledgebase.
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        JSON string with top 25 matching document chunks
    """
    embedder = _get_embedder()
    
    # Get query embedding
    query_embedding = await embedder.embed(query)
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
    
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT 
                content,
                metadata,
                1 - (embedding <=> $1::vector) as similarity
            FROM documents_pg
            ORDER BY embedding <=> $1::vector
            LIMIT 25
            """,
            embedding_str
        )
    
    results = []
    for row in rows:
        results.append({
            "content": row["content"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "similarity": float(row["similarity"])
        })
    
    return json.dumps(results, indent=2)
```

- **PATTERN**: Embedding from `embedder.py`, DB from `connection.py`
- **GOTCHA**: pgvector uses `<=>` for cosine distance, similarity = 1 - distance
- **VALIDATE**: `python -c "from src.agent.tools.vector_search import vector_search_tool; print(vector_search_tool.name)"`

---

### CREATE `src/agent/tools/list_docs.py`

**IMPLEMENT**: List available documents from document_metadata

```python
"""List documents tool."""
import json
from langchain.tools import tool

from src.database import get_connection


@tool
async def list_docs_tool() -> str:
    """List all available documents in the knowledge base with their metadata.
    
    Returns:
        JSON string with all documents and their schemas
    """
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT 
                id,
                filename,
                file_type,
                metadata,
                created_at,
                schema
            FROM document_metadata
            ORDER BY created_at DESC
            """
        )
    
    documents = []
    for row in rows:
        documents.append({
            "file_id": row["id"],
            "filename": row["filename"],
            "file_type": row["file_type"],
            "schema": row["schema"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None
        })
    
    return json.dumps(documents, indent=2)
```

- **PATTERN**: SQL pattern from n8n workflow
- **VALIDATE**: `python -c "from src.agent.tools.list_docs import list_docs_tool; print(list_docs_tool.name)"`

---

### CREATE `src/agent/tools/get_content.py`

**IMPLEMENT**: Get full document content by file_id

```python
"""Get document content tool."""
import json
from langchain.tools import tool

from src.database import get_connection


@tool
async def get_content_tool(file_id: str) -> str:
    """Get the complete text content of a specific document by its file_id.
    
    Args:
        file_id: The unique identifier of the document
        
    Returns:
        Complete document text (all chunks concatenated)
    """
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT content, metadata
            FROM documents_pg
            WHERE metadata->>'file_id' = $1
            ORDER BY (metadata->>'chunk_index')::int
            """,
            file_id
        )
    
    if not rows:
        return f"No document found with file_id: {file_id}"
    
    # Concatenate all chunks
    full_content = "\n".join(row["content"] for row in rows)
    
    # Get metadata from first chunk
    metadata = json.loads(rows[0]["metadata"]) if rows[0]["metadata"] else {}
    
    result = {
        "file_id": file_id,
        "title": metadata.get("file_title", "Unknown"),
        "total_chunks": len(rows),
        "content": full_content
    }
    
    return json.dumps(result, indent=2)
```

- **PATTERN**: JSONB access with `->>'key'`
- **VALIDATE**: `python -c "from src.agent.tools.get_content import get_content_tool; print(get_content_tool.name)"`

---

### CREATE `src/agent/tools/sql_query.py`

**IMPLEMENT**: Execute SQL queries against tabular data

```python
"""SQL query tool for tabular data."""
import json
from langchain.tools import tool

from src.database import get_connection


@tool
async def sql_query_tool(sql_query: str, dataset_id: str) -> str:
    """Execute a SQL query against tabular data stored in document_rows.
    
    The document_rows table has columns: id, dataset_id, row_data (JSONB).
    Use JSONB operators to access columns, e.g., row_data->>'column_name'.
    
    Args:
        sql_query: The SQL query with JSONB operators for column access
        dataset_id: The file_id of the dataset to query
        
    Returns:
        JSON string with query results or error message
    """
    # Security: Basic SQL injection prevention
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
    sql_upper = sql_query.upper()
    for word in forbidden:
        if word in sql_upper:
            return json.dumps({"error": f"Forbidden SQL operation: {word}"})
    
    # Ensure query is scoped to the dataset
    if "document_rows" not in sql_query.lower():
        return json.dumps({"error": "Query must reference document_rows table"})
    
    try:
        async with get_connection() as conn:
            rows = await conn.fetch(sql_query)
        
        results = [dict(row) for row in rows]
        
        # Convert any non-serializable types
        for result in results:
            for key, value in result.items():
                if hasattr(value, 'isoformat'):
                    result[key] = value.isoformat()
        
        return json.dumps({
            "row_count": len(results),
            "results": results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e)})
```

- **PATTERN**: Security checks for SQL queries
- **GOTCHA**: JSONB operators: `->>` for text, `->` for JSON
- **VALIDATE**: `python -c "from src.agent.tools.sql_query import sql_query_tool; print(sql_query_tool.name)"`

---

### CREATE `src/agent/agent.py`

**IMPLEMENT**: Main RAG agent using LangChain

```python
"""Main RAG AI Agent implementation."""
import asyncio
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from src.config import get_settings
from src.agent.memory import PostgresChatMemory
from src.agent.prompts import RAG_SYSTEM_PROMPT
from src.agent.tools import (
    vector_search_tool,
    list_docs_tool,
    get_content_tool,
    sql_query_tool,
)


@dataclass
class AgentResponse:
    """Response from the agent."""
    content: str
    tool_calls: list = None


class RAGAgent:
    """RAG AI Agent with tools and memory.
    
    Matches the n8n "RAG AI Agent" node behavior.
    """
    
    def __init__(self, session_id: str):
        """Initialize the agent.
        
        Args:
            session_id: Unique session ID for memory
        """
        settings = get_settings()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.7,
        )
        
        # Initialize memory
        self.memory = PostgresChatMemory(session_id=session_id, max_messages=10)
        
        # Define tools
        self.tools = [
            vector_search_tool,
            list_docs_tool,
            get_content_tool,
            sql_query_tool,
        ]
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    async def _get_chat_history(self) -> list:
        """Convert memory to LangChain message format."""
        messages = await self.memory.get_messages()
        history = []
        for msg in messages:
            if msg.role == "human":
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))
        return history
    
    async def invoke(self, user_input: str) -> AgentResponse:
        """Process a user message and return a response.
        
        Args:
            user_input: The user's message
            
        Returns:
            AgentResponse with the AI response
        """
        # Get chat history
        chat_history = await self._get_chat_history()
        
        # Save user message
        await self.memory.add_message("human", user_input)
        
        # Run in executor (LangChain sync)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
        )
        
        response_content = result.get("output", "")
        
        # Save AI response
        await self.memory.add_message("ai", response_content)
        
        return AgentResponse(content=response_content)


async def create_agent(session_id: str) -> RAGAgent:
    """Factory function to create an agent.
    
    Args:
        session_id: Unique session ID
        
    Returns:
        Configured RAGAgent instance
    """
    return RAGAgent(session_id=session_id)
```

- **PATTERN**: LangChain agent patterns from Context7 docs
- **IMPORTS**: All tools, memory, prompts
- **VALIDATE**: `python -c "from src.agent.agent import RAGAgent; print('OK')"`

---

### UPDATE `src/agent/__init__.py`

**IMPLEMENT**: Export agent and tools

```python
"""RAG Agent module."""
from .agent import RAGAgent, create_agent, AgentResponse
from .memory import PostgresChatMemory, ChatMessage
from .prompts import RAG_SYSTEM_PROMPT

__all__ = [
    "RAGAgent",
    "create_agent",
    "AgentResponse",
    "PostgresChatMemory",
    "ChatMessage",
    "RAG_SYSTEM_PROMPT",
]
```

- **VALIDATE**: `python -c "from src.agent import RAGAgent, create_agent; print('OK')"`

---

### CREATE `tests/test_agent.py`

**IMPLEMENT**: Unit tests for agent components

```python
"""Tests for the RAG agent and tools."""
import pytest
import json
from unittest.mock import patch, AsyncMock

from src.agent.memory import PostgresChatMemory, ChatMessage
from src.agent.prompts import RAG_SYSTEM_PROMPT


class TestChatMemory:
    """Tests for PostgresChatMemory."""
    
    @pytest.mark.asyncio
    async def test_add_message(self, mock_db_connection):
        """Test adding a message to memory."""
        memory = PostgresChatMemory(session_id="test-session")
        await memory.add_message("human", "Hello")
        # Verify DB was called
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
        from datetime import datetime
        mock_db_connection.fetch.return_value = [
            {"role": "ai", "content": "Hi!", "created_at": datetime(2024, 1, 1, 12, 1)},
            {"role": "human", "content": "Hello", "created_at": datetime(2024, 1, 1, 12, 0)},
        ]
        memory = PostgresChatMemory(session_id="test-session")
        messages = await memory.get_messages()
        # Should be reversed (oldest first)
        assert messages[0].role == "human"
        assert messages[1].role == "ai"


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


class TestVectorSearchTool:
    """Tests for vector search tool."""
    
    @pytest.mark.asyncio
    async def test_vector_search_returns_json(self, mock_db_connection, mock_embedder):
        """Test vector search returns valid JSON."""
        from src.agent.tools.vector_search import vector_search_tool
        
        mock_db_connection.fetch.return_value = [
            {"content": "test content", "metadata": '{"file_id": "123"}', "similarity": 0.9}
        ]
        
        result = await vector_search_tool.coroutine("test query")
        parsed = json.loads(result)
        
        assert len(parsed) == 1
        assert parsed[0]["content"] == "test content"


class TestSqlQueryTool:
    """Tests for SQL query tool."""
    
    @pytest.mark.asyncio
    async def test_blocks_drop_statement(self, mock_db_connection):
        """Test that DROP statements are blocked."""
        from src.agent.tools.sql_query import sql_query_tool
        
        result = await sql_query_tool.coroutine(
            sql_query="DROP TABLE document_rows",
            dataset_id="test"
        )
        parsed = json.loads(result)
        
        assert "error" in parsed
        assert "DROP" in parsed["error"]
    
    @pytest.mark.asyncio
    async def test_requires_document_rows_table(self, mock_db_connection):
        """Test that query must reference document_rows."""
        from src.agent.tools.sql_query import sql_query_tool
        
        result = await sql_query_tool.coroutine(
            sql_query="SELECT * FROM users",
            dataset_id="test"
        )
        parsed = json.loads(result)
        
        assert "error" in parsed
```

- **PATTERN**: Test patterns from `tests/test_pipeline.py`
- **VALIDATE**: `pytest tests/test_agent.py -v --co`

---

## TESTING STRATEGY

### Unit Tests

- Test each tool in isolation with mocked database
- Test memory add/get operations
- Test system prompt content

### Integration Tests

1. **Agent invocation test**: Send a query, verify response
2. **Memory persistence test**: Multiple queries, verify history
3. **Tool selection test**: Query types trigger correct tools

### Edge Cases

- Empty database (no documents)
- Invalid file_id for get_content
- Malicious SQL in sql_query
- Very long queries (token limits)

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
ruff check src/agent/ --fix
ruff format src/agent/
```

### Level 2: Unit Tests

```bash
pytest tests/test_agent.py -v
```

### Level 3: Import Tests

```bash
python -c "from src.agent import RAGAgent, create_agent; print('All imports OK')"
python -c "from src.agent.tools import vector_search_tool, list_docs_tool, get_content_tool, sql_query_tool; print('All tools OK')"
```

### Level 4: Manual Validation

```python
# In Python REPL
import asyncio
from src.agent import create_agent

async def test():
    agent = await create_agent("test-session")
    response = await agent.invoke("What documents do I have?")
    print(response.content)

asyncio.run(test())
```

---

## ACCEPTANCE CRITERIA

- [ ] Vector search returns top 25 similar documents
- [ ] List docs shows all documents with metadata
- [ ] Get content returns full document by file_id
- [ ] SQL query executes safely against document_rows
- [ ] Chat memory persists across invocations
- [ ] Agent uses correct tool based on query type
- [ ] System prompt matches n8n exactly
- [ ] All validation commands pass

---

## COMPLETION CHECKLIST

- [ ] All files created in order
- [ ] Each validation passed
- [ ] Unit tests pass
- [ ] Import tests pass
- [ ] Ruff linting passes
- [ ] Manual test with agent.invoke() works

---

## NOTES

### Deferred: Re-ranker Tool

The Google Discovery Engine re-ranker requires a GCP Service Account with proper permissions. This is **optional** and can be added later:

```python
# Future: src/agent/tools/reranker.py
# Uses HTTP POST to:
# https://discoveryengine.googleapis.com/v1/projects/{PROJECT}/locations/global/rankingConfigs/default_ranking_config:rank
# Body: { "model": "semantic-ranker-512@latest", "query": "...", "records": [...], "topN": 5 }
```

### Model Configuration

From n8n workflow (matching exactly):
- Chat: `models/gemini-2.5-pro`
- Embeddings: `models/gemini-embedding-001` (configured in `embedder.py`)
- Context window: 10 messages (matching `PostgresChatMemory.max_messages`)

### Confidence Score: 8/10

High confidence because:
- Patterns well-documented in LangChain
- n8n workflow provides exact specifications
- Existing codebase has all necessary infrastructure
- Only risk is async/sync compatibility between LangChain and asyncpg
