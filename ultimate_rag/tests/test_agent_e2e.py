"""Test the full RAG agent end-to-end."""
import asyncio
from src.agent import create_agent
from src.database import close_pool


async def main():
    print("=" * 50)
    print("TESTING FULL RAG AGENT - END TO END")
    print("=" * 50)
    
    try:
        # Create agent with a test session
        agent = await create_agent("live-e2e-test-session")
        print("✓ Agent created successfully")
        
        # Test 1: Ask about documents
        print("\n--- Query 1: What documents do I have? ---")
        response = await agent.invoke("What documents are available in my knowledge base?")
        print(f"Response: {response.content[:500]}...")
        
        # Test 2: RAG search
        print("\n--- Query 2: Search for test content ---")
        response = await agent.invoke("What is mentioned in the test documents?")
        print(f"Response: {response.content[:500]}...")
        
        print("\n" + "=" * 50)
        print("FULL AGENT TEST COMPLETE")
        print("=" * 50)
        
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
