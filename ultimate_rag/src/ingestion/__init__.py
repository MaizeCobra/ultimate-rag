"""Ingestion module for file processing.

Provides a complete pipeline for ingesting various file types,
chunking content, generating embeddings, and storing in the database.
"""
from .file_detector import (
    FileCategory,
    FileInfo,
    detect_file_type,
    is_supported,
)
from .chunker import TextChunker, Chunk
from .embedder import Embedder
from .pipeline import IngestPipeline, ingest_file
from .processors.base import BaseProcessor, ProcessedDocument

__all__ = [
    # File detection
    "FileCategory",
    "FileInfo",
    "detect_file_type",
    "is_supported",
    # Chunking
    "TextChunker",
    "Chunk",
    # Embedding
    "Embedder",
    # Pipeline
    "IngestPipeline",
    "ingest_file",
    # Processors
    "BaseProcessor",
    "ProcessedDocument",
]
