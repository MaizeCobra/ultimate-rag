"""PostgreSQL-based chat memory for conversation history."""

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
                self.session_id,
                role,
                content,
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
                self.session_id,
                self.max_messages,
            )

        # Reverse to get chronological order
        messages = [
            ChatMessage(
                role=r["role"], content=r["content"], created_at=r["created_at"]
            )
            for r in reversed(rows)
        ]
        return messages

    async def clear(self) -> None:
        """Clear all messages for this session."""
        async with get_connection() as conn:
            await conn.execute(
                "DELETE FROM chat_memory WHERE session_id = $1", self.session_id
            )
