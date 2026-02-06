"""PostgreSQL database connection management."""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import asyncpg
from asyncpg import Pool, Connection

from src.config import get_settings


_pool: Pool | None = None


async def get_pool() -> Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            host="localhost",
            port=5433,
            min_size=2,
            max_size=10,
        )
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection, None]:
    """Get a database connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_schema() -> None:
    """Initialize database schema from schema.sql."""
    schema_path = Path(__file__).parent / "schema.sql"
    schema_sql = schema_path.read_text()
    
    async with get_connection() as conn:
        await conn.execute(schema_sql)


async def check_tables_exist() -> dict[str, bool]:
    """Check if all required tables exist."""
    tables = ["documents_pg", "document_metadata", "document_rows", "chat_memory"]
    result = {}
    
    async with get_connection() as conn:
        for table in tables:
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
                """,
                table
            )
            result[table] = exists
    
    return result
