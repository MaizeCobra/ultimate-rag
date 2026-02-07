"""Check chat memory entries."""
import asyncio
from src.database import get_connection, close_pool


async def check():
    async with get_connection() as conn:
        rows = await conn.fetch(
            "SELECT session_id, role, content, created_at FROM chat_memory ORDER BY created_at DESC LIMIT 10"
        )
        print(f"Chat memory entries: {len(rows)}\n")
        for r in rows:
            content_preview = str(r["content"])[:100].replace("\n", " ")
            print(f"[{r['session_id'][:20]}] {r['role']:5} : {content_preview}...")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(check())
