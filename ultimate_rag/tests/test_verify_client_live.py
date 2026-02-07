"""Live verification of RAGClient with existing data."""
import asyncio
import logging
from src.client import RAGClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_client():
    print("\n🚀 Starting RAGClient Verification...\n")
    
    async with RAGClient() as client:
        # 1. Test Stats
        print("📊 1. Fetching Database Stats...")
        stats = await client.get_stats()
        print(f"   Documents: {stats.get('documents_pg', 'N/A')}")
        print(f"   Metadata:  {stats.get('document_metadata', 'N/A')}")
        print(f"   Duplicates: {stats.get('duplicate_files', 'N/A')}")
        
        # 2. Test Search (Direct Vector)
        print("\n🔍 2. Testing Direct Vector Search...")
        query = "What is the capital of France?" # Generic query to test flow
        try:
            results = await client.search(query, limit=2)
            print(f"   ✅ Search successful! (Result length: {len(results)})")
            # print(f"   Sample: {results[:100]}...") 
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

        # 3. Test Query (Agent)
        print("\n🤖 3. Testing Agent Query (Conversation)...")
        # We know we have some docs, let's ask a generic question or specific if we knew content
        # "Summarize the documents" is a good test
        q = "List the documents you have access to."
        try:
            response = await client.query(q)
            print(f"   User: {q}")
            print(f"   AI:   {response}")
            print("   ✅ Agent query successful!")
        except Exception as e:
            print(f"   ❌ Agent query failed: {e}")

        # 4. Test Management (Cleanup - Safe)
        print("\n🧹 4. Testing Cleanup (Dry/Safe)...")
        # We already cleaned up in Phase 3.5, so expecting 0 changes
        try:
            cleanup_stats = await client.cleanup()
            print(f"   Duplicates removed: {cleanup_stats['duplicates_removed']}")
            print(f"   Orphans removed:    {cleanup_stats['orphans_removed']}")
            print("   ✅ Cleanup command successful!")
        except Exception as e:
            print(f"   ❌ Cleanup failed: {e}")

    print("\n✨ Verification Complete!")

if __name__ == "__main__":
    asyncio.run(verify_client())
