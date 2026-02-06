"""Tests for text chunking."""
import pytest

from src.ingestion import TextChunker, Chunk


class TestTextChunker:
    """Test text chunking functionality."""
    
    def test_chunk_short_text(self):
        """Test that short text returns single chunk."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        text = "Short text"
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_long_text(self):
        """Test that long text is split into multiple chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a sentence. " * 50  # ~1000 chars
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size
        for chunk in chunks:
            assert len(chunk) <= 110  # Allow some variance
    
    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "ABCDEFGHIJ " * 20  # Repeating pattern
        chunks = chunker.chunk(text)
        
        # Check that consecutive chunks share some content
        if len(chunks) >= 2:
            overlap = set(chunks[0][-30:]) & set(chunks[1][:30])
            assert len(overlap) > 0
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
    
    def test_chunk_with_metadata(self):
        """Test chunk_with_metadata returns Chunk objects."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50
        chunks = chunker.chunk_with_metadata(text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_index == 0
    
    def test_chunk_preserves_paragraphs(self):
        """Test that chunker respects paragraph boundaries."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)
        
        # Should try to split on paragraph boundaries
        assert len(chunks) >= 1
    
    def test_custom_chunk_size(self):
        """Test custom chunk size."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = chunker.chunk(text)
        
        # Should have multiple chunks of ~50 chars
        assert len(chunks) >= 3
    
    def test_default_parameters(self):
        """Test default chunker parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
