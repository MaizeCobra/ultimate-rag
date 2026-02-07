"""Check for duplicate documents in the database."""
import asyncio
from src.database import get_connection, close_pool


async def check():
    async with get_connection() as conn:
        rows = await conn.fetch(
            "SELECT id, filename, created_at FROM document_metadata ORDER BY created_at"
        )
        print(f"Total documents: {len(rows)}\n")
        for r in rows:
            print(f"{r['filename']:20} | {str(r['id'])[:8]}... | {r['created_at']}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(check())
