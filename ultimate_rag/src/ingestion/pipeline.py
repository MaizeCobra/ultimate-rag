"""Ingestion pipeline orchestrator.

Coordinates file detection, processing, chunking, embedding,
and storage for all supported file types.
"""
import logging
from pathlib import Path
from typing import Any

from src.database import get_connection

from .file_detector import FileCategory, detect_file_type
from .chunker import TextChunker
from .embedder import Embedder
from .processors.base import ProcessedDocument
from .processors.text import TextProcessor
from .processors.json_processor import JsonProcessor
from .processors.spreadsheet import SpreadsheetProcessor
from .processors.image import ImageProcessor
from .processors.audio import AudioProcessor
from .processors.video import VideoProcessor
from .processors.pdf import PDFProcessor


logger = logging.getLogger(__name__)


class IngestPipeline:
    """Orchestrates the full ingestion pipeline.
    
    Flow: detect → process → chunk → embed → store
    
    Attributes:
        chunker: TextChunker for splitting content
        embedder: Embedder for generating vectors
        processors: Dict mapping file categories to processors
    """
    
    def __init__(self):
        """Initialize pipeline with all components."""
        self.chunker = TextChunker()
        self.embedder = Embedder()
        
        # Initialize processors for each file type
        self.processors = {
            FileCategory.TEXT: TextProcessor(),
            FileCategory.JSON: JsonProcessor(),
            FileCategory.CSV: SpreadsheetProcessor(),
            FileCategory.XLSX: SpreadsheetProcessor(),
            FileCategory.IMAGE: ImageProcessor(),
            FileCategory.AUDIO: AudioProcessor(),
            FileCategory.VIDEO: VideoProcessor(),
            FileCategory.PDF: PDFProcessor(),
        }
    
    async def ingest(self, file_path: str | Path) -> str:
        """Ingest a file and return its document ID.
        
        Args:
            file_path: Path to the file to ingest
            
        Returns:
            Document ID of the ingested file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 1. Detect file type
        file_info = detect_file_type(file_path)
        logger.info(f"Detected file type: {file_info.category.value} for {file_path.name}")
        
        if file_info.category == FileCategory.UNKNOWN:
            raise ValueError(f"Unsupported file type: {file_info.extension}")
        
        # 2. Process with appropriate processor
        processor = self.processors[file_info.category]
        doc = processor.process(file_path)
        logger.info(f"Processed document: {doc.title} ({len(doc.content)} chars)")
        
        # 3. Store metadata first
        await self._store_metadata(doc)
        
        # 4. Store rows if tabular data
        if doc.rows:
            await self._store_rows(doc.file_id, doc.rows)
            logger.info(f"Stored {len(doc.rows)} rows for {doc.title}")
        
        # 5. Chunk content (skip for short content)
        if len(doc.content) < 100:
            chunks = [doc.content] if doc.content.strip() else []
        else:
            chunks = self.chunker.chunk(doc.content)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # 6. Embed and store each chunk
        for i, chunk in enumerate(chunks):
            embedding = await self.embedder.embed(chunk)
            await self._store_document(doc.file_id, chunk, embedding, doc.metadata, i)
        
        logger.info(f"Ingestion complete: {doc.file_id}")
        return doc.file_id
    
    async def _store_metadata(self, doc: ProcessedDocument) -> None:
        """Store document metadata.
        
        Args:
            doc: Processed document
        """
        async with get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO document_metadata (id, filename, file_type, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    file_type = EXCLUDED.file_type,
                    metadata = EXCLUDED.metadata
                """,
                doc.file_id,
                doc.title,
                doc.metadata.get("type", "unknown"),
                str(doc.metadata).replace("'", '"'),  # Convert to JSON string
            )
    
    async def _store_rows(self, dataset_id: str, rows: list[dict]) -> None:
        """Store tabular data rows.
        
        Args:
            dataset_id: Document ID
            rows: List of row dictionaries
        """
        import json
        async with get_connection() as conn:
            for row in rows:
                await conn.execute(
                    """
                    INSERT INTO document_rows (dataset_id, row_data)
                    VALUES ($1, $2::jsonb)
                    """,
                    dataset_id,
                    json.dumps(row),
                )
    
    async def _store_document(
        self,
        file_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any],
        chunk_index: int,
    ) -> None:
        """Store a document chunk with embedding.
        
        Args:
            file_id: Document ID
            content: Chunk content
            embedding: Embedding vector
            metadata: Document metadata
            chunk_index: Index of this chunk
        """
        import json
        
        # Add chunk info to metadata
        chunk_metadata = {
            **metadata,
            "chunk_index": chunk_index,
        }
        
        async with get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO documents_pg (content, metadata, embedding)
                VALUES ($1, $2::jsonb, $3::vector)
                """,
                content,
                json.dumps(chunk_metadata),
                f"[{','.join(str(x) for x in embedding)}]",
            )


async def ingest_file(file_path: str | Path) -> str:
    """Convenience function to ingest a single file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Document ID
    """
    pipeline = IngestPipeline()
    return await pipeline.ingest(file_path)
