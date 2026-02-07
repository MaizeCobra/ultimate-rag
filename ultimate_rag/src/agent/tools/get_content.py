"""Get document content tool."""

import json

from langchain_core.tools import tool

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
            file_id,
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
        "content": full_content,
    }

    return json.dumps(result, indent=2)
