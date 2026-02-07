"""Live integration test for Phase 3 Agent tools."""
import asyncio
import json
from src.database import get_connection, close_pool
from src.agent.tools import (
    list_docs_tool,
    vector_search_tool,
    get_content_tool,
    sql_query_tool,
)


async def test_list_docs():
    """Test list_docs_tool."""
    print("\n=== TEST: list_docs_tool ===")
    result = await list_docs_tool.ainvoke({})
    docs = json.loads(result)
    print(f"✓ Found {len(docs)} documents")
    for doc in docs[:3]:
        print(f"  - {doc['filename']} ({doc['file_type']})")
    return docs


async def test_vector_search():
    """Test vector_search_tool."""
    print("\n=== TEST: vector_search_tool ===")
    result = await vector_search_tool.ainvoke({"query": "test document"})
    chunks = json.loads(result)
    print(f"✓ Found {len(chunks)} chunks")
    for chunk in chunks[:3]:
        print(f"  - similarity: {chunk['similarity']:.4f}")
        print(f"    content: {chunk['content'][:60]}...")
    return chunks


async def test_get_content():
    """Test get_content_tool."""
    print("\n=== TEST: get_content_tool ===")
    
    # First get a file_id from document_metadata
    async with get_connection() as conn:
        row = await conn.fetchrow("SELECT id FROM document_metadata LIMIT 1")
    
    if row:
        file_id = row["id"]
        print(f"Testing with file_id: {file_id}")
        result = await get_content_tool.ainvoke({"file_id": file_id})
        data = json.loads(result)
        if "error" in str(result).lower() or "No document" in result:
            print(f"✗ Error: {result}")
            return None
        print(f"✓ Got content for file: {data.get('title', 'Unknown')}")
        print(f"  Total chunks: {data.get('total_chunks', 0)}")
        print(f"  Content preview: {data.get('content', '')[:100]}...")
        return data
    else:
        print("✗ No documents found in document_metadata")
        return None


async def test_sql_query():
    """Test sql_query_tool."""
    print("\n=== TEST: sql_query_tool ===")
    
    # Check if there's data in document_rows
    async with get_connection() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM document_rows")
    
    if count > 0:
        query = "SELECT COUNT(*) as total FROM document_rows"
        result = await sql_query_tool.ainvoke({
            "sql_query": query,
            "dataset_id": "test"
        })
        data = json.loads(result)
        if "error" in data:
            print(f"✗ Error: {data['error']}")
        else:
            print(f"✓ Query returned {data['row_count']} rows")
            print(f"  Results: {data['results']}")
        return data
    else:
        print("✗ No rows in document_rows - skipping SQL test")
        return None


async def test_sql_security():
    """Test SQL injection prevention."""
    print("\n=== TEST: SQL Security ===")
    
    # Test DROP
    result = await sql_query_tool.ainvoke({
        "sql_query": "DROP TABLE document_rows",
        "dataset_id": "test"
    })
    data = json.loads(result)
    if "error" in data and "DROP" in data["error"]:
        print("✓ DROP statement blocked correctly")
    else:
        print("✗ DROP statement NOT blocked!")
    
    # Test DELETE
    result = await sql_query_tool.ainvoke({
        "sql_query": "DELETE FROM document_rows",
        "dataset_id": "test"
    })
    data = json.loads(result)
    if "error" in data and "DELETE" in data["error"]:
        print("✓ DELETE statement blocked correctly")
    else:
        print("✗ DELETE statement NOT blocked!")


async def main():
    print("=" * 50)
    print("PHASE 3 LIVE INTEGRATION TESTS")
    print("=" * 50)
    
    try:
        await test_list_docs()
        await test_vector_search()
        await test_get_content()
        await test_sql_query()
        await test_sql_security()
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETE")
        print("=" * 50)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
