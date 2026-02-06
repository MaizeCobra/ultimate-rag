"""Integration tests for the ingestion pipeline."""
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.database import get_connection, init_schema, close_pool
from src.ingestion import IngestPipeline, detect_file_type, FileCategory


class TestIngestPipeline:
    """Integration tests for the ingestion pipeline."""
    
    @pytest.mark.asyncio
    async def test_ingest_text_file(self, sample_text_file):
        """Test ingesting a text file."""
        # Mock the embedder to avoid API calls
        with patch("src.ingestion.pipeline.Embedder") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 3072)
            mock_embedder_class.return_value = mock_embedder
            
            pipeline = IngestPipeline()
            pipeline.embedder = mock_embedder
            
            doc_id = await pipeline.ingest(sample_text_file)
            
            assert doc_id is not None
            assert len(doc_id) == 36  # UUID length
    
    @pytest.mark.asyncio
    async def test_ingest_json_file(self, sample_json_file):
        """Test ingesting a JSON file."""
        with patch("src.ingestion.pipeline.Embedder") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 3072)
            mock_embedder_class.return_value = mock_embedder
            
            pipeline = IngestPipeline()
            pipeline.embedder = mock_embedder
            
            doc_id = await pipeline.ingest(sample_json_file)
            
            assert doc_id is not None
    
    @pytest.mark.asyncio
    async def test_ingest_csv_file(self, sample_csv_file):
        """Test ingesting a CSV file stores rows."""
        with patch("src.ingestion.pipeline.Embedder") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 3072)
            mock_embedder_class.return_value = mock_embedder
            
            pipeline = IngestPipeline()
            pipeline.embedder = mock_embedder
            
            doc_id = await pipeline.ingest(sample_csv_file)
            
            # Verify rows were stored
            async with get_connection() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM document_rows WHERE dataset_id = $1",
                    doc_id
                )
                assert count == 3  # 3 rows in sample CSV
    
    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self):
        """Test that ingesting non-existent file raises error."""
        pipeline = IngestPipeline()
        
        with pytest.raises(FileNotFoundError):
            await pipeline.ingest(Path("nonexistent.txt"))
    
    @pytest.mark.asyncio
    async def test_ingest_unsupported_file(self, tmp_path):
        """Test that ingesting unsupported file raises error."""
        pipeline = IngestPipeline()
        unsupported = tmp_path / "data.xyz"
        unsupported.write_text("some data")
        
        with pytest.raises(ValueError, match="Unsupported"):
            await pipeline.ingest(unsupported)
    
    @pytest.mark.asyncio
    async def test_documents_stored_in_database(self, sample_text_file):
        """Test that documents are stored in database."""
        with patch("src.ingestion.pipeline.Embedder") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 3072)
            mock_embedder_class.return_value = mock_embedder
            
            pipeline = IngestPipeline()
            pipeline.embedder = mock_embedder
            
            doc_id = await pipeline.ingest(sample_text_file)
            
            # Verify document chunks are in database
            async with get_connection() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM documents_pg WHERE metadata->>'file_id' = $1",
                    doc_id
                )
                assert count > 0
    
    @pytest.mark.asyncio
    async def test_metadata_stored_in_database(self, sample_text_file):
        """Test that metadata is stored in database."""
        with patch("src.ingestion.pipeline.Embedder") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = AsyncMock(return_value=[0.1] * 3072)
            mock_embedder_class.return_value = mock_embedder
            
            pipeline = IngestPipeline()
            pipeline.embedder = mock_embedder
            
            doc_id = await pipeline.ingest(sample_text_file)
            
            # Verify metadata is stored
            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM document_metadata WHERE id = $1",
                    doc_id
                )
                assert row is not None
                assert row["filename"] == sample_text_file.name


class TestPipelineFileTypeRouting:
    """Test that pipeline routes files to correct processors."""
    
    def test_text_file_routing(self, sample_text_file):
        """Test text file is routed correctly."""
        info = detect_file_type(sample_text_file)
        assert info.category == FileCategory.TEXT
    
    def test_json_file_routing(self, sample_json_file):
        """Test JSON file is routed correctly."""
        info = detect_file_type(sample_json_file)
        assert info.category == FileCategory.JSON
    
    def test_csv_file_routing(self, sample_csv_file):
        """Test CSV file is routed correctly."""
        info = detect_file_type(sample_csv_file)
        assert info.category == FileCategory.CSV
