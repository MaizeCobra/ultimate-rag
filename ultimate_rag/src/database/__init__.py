"""Database module exports."""
from .connection import (
    get_pool,
    get_connection,
    close_pool,
    init_schema,
    check_tables_exist,
)
from .management import (
    delete_document,
    document_exists,
    update_document,
    get_duplicate_files,
    cleanup_duplicates,
    cleanup_orphan_chunks,
    get_stats,
    truncate_all_tables,
)

__all__ = [
    # Connection
    "get_pool",
    "get_connection", 
    "close_pool",
    "init_schema",
    "check_tables_exist",
    # Management
    "delete_document",
    "document_exists",
    "update_document",
    "get_duplicate_files",
    "cleanup_duplicates",
    "cleanup_orphan_chunks",
    "get_stats",
    "truncate_all_tables",
]
