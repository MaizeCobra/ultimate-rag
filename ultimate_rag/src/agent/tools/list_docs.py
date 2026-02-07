"""List documents tool."""

import json

from langchain_core.tools import tool

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
        documents.append(
            {
                "file_id": row["id"],
                "filename": row["filename"],
                "file_type": row["file_type"],
                "schema": row["schema"],
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
            }
        )

    return json.dumps(documents, indent=2)
