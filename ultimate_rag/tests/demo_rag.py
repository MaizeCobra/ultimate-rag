"""Demo: Ingest PDF and ask the agent about it."""
import asyncio
from pathlib import Path

from src.ingestion import ingest_file
from src.agent import create_agent
from src.database import close_pool


async def main():
    pdf_path = Path("tests/fixtures/RPA - LAB5 - 23BRS1129.pdf")
    
    print("=" * 60)
    print("DEMO: Ingest PDF and Query Agent")
    print("=" * 60)
    
    # Step 1: Ingest the PDF
    print(f"\n[1] Ingesting: {pdf_path.name}")
    file_id = await ingest_file(str(pdf_path))
    print(f"✓ Ingested! file_id: {file_id}")
    
    # Step 2: Create agent and ask
    print("\n[2] Creating agent...")
    agent = await create_agent("pdf-demo-session")
    
    print("\n[3] Asking: 'What's in my documents?'")
    print("-" * 60)
    response = await agent.invoke("What's in my documents?")
    print(response.content)
    print("-" * 60)
    
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
