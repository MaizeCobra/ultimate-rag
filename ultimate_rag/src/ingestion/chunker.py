"""Text chunking for the ingestion pipeline.

Uses LangChain's RecursiveCharacterTextSplitter to split text
into chunks suitable for embedding and retrieval.
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """A chunk of text with metadata.
    
    Attributes:
        text: The chunk text content
        start_index: Starting character position in original text
        chunk_index: Index of this chunk (0-based)
    """
    text: str
    start_index: int
    chunk_index: int


class TextChunker:
    """Splits text into overlapping chunks for embedding.
    
    Uses RecursiveCharacterTextSplitter which recursively splits
    text on common separators (paragraphs, sentences, words) until
    each chunk fits within the size limit.
    
    Attributes:
        chunk_size: Maximum characters per chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        """Initialize the text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            separators: Custom separators (defaults to newlines, spaces)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunk strings
            
        Example:
            >>> chunker = TextChunker(chunk_size=100, chunk_overlap=20)
            >>> chunks = chunker.chunk("Long text here...")
            >>> len(chunks)
            3
        """
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text)
    
    def chunk_with_metadata(self, text: str) -> list[Chunk]:
        """Split text into chunks with metadata.
        
        Args:
            text: Text to split
            
        Returns:
            List of Chunk objects with position information
        """
        if not text or not text.strip():
            return []
        
        # Use Document to get start_index metadata
        from langchain_core.documents import Document
        doc = Document(page_content=text)
        split_docs = self.splitter.split_documents([doc])
        
        return [
            Chunk(
                text=doc.page_content,
                start_index=doc.metadata.get("start_index", 0),
                chunk_index=i,
            )
            for i, doc in enumerate(split_docs)
        ]
