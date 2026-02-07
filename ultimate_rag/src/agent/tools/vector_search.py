"""Vector search tool using pgvector."""

import json

from langchain_core.tools import tool

from src.database import get_connection
from src.ingestion.embedder import Embedder


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
            embedding_str,
        )

    results = []
    for row in rows:
        results.append(
            {
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "similarity": float(row["similarity"]),
            }
        )

    return json.dumps(results, indent=2)
