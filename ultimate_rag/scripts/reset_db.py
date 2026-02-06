import asyncio
from src.database import get_pool, close_pool

async def reset():
    print("Resetting database...")
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            DROP TABLE IF EXISTS documents_pg CASCADE;
            DROP TABLE IF EXISTS document_rows CASCADE;
            DROP TABLE IF EXISTS document_metadata CASCADE;
            DROP TABLE IF EXISTS chat_memory CASCADE;
        """)
        print("Tables dropped successfully.")
    await close_pool()

if __name__ == "__main__":
    asyncio.run(reset())
