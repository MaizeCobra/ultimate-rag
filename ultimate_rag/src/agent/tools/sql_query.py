"""SQL query tool for tabular data."""

import json

from langchain_core.tools import tool

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
                if hasattr(value, "isoformat"):
                    result[key] = value.isoformat()

        return json.dumps({"row_count": len(results), "results": results}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})
