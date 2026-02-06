"""Base processor for the ingestion pipeline.

All file processors inherit from BaseProcessor and implement
the process() method to extract text content from files.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcessedDocument:
    """Result of processing a file.
    
    Attributes:
        file_id: Unique identifier for the document
        title: Display title (usually filename)
        content: Extracted text content for chunking/embedding
        metadata: Additional metadata (file type, source, etc.)
        rows: For tabular data, list of row dictionaries
    """
    file_id: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] | None = None
    
    def __post_init__(self):
        """Ensure metadata always has required fields."""
        if "file_id" not in self.metadata:
            self.metadata["file_id"] = self.file_id
        if "title" not in self.metadata:
            self.metadata["title"] = self.title


class BaseProcessor(ABC):
    """Abstract base class for file processors.
    
    All processors must implement the process() method which
    takes a file path and returns a ProcessedDocument.
    """
    
    @abstractmethod
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process a file and extract text content.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            ProcessedDocument with extracted content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        pass
    
    def validate_file(self, file_path: Path) -> None:
        """Validate that the file exists and is accessible.
        
        Args:
            file_path: Path to validate
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file isn't readable
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
