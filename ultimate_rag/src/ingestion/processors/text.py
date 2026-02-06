"""Text and Markdown file processor."""
import uuid
from pathlib import Path

from .base import BaseProcessor, ProcessedDocument


class TextProcessor(BaseProcessor):
    """Processor for plain text and markdown files.
    
    Reads the file content directly and returns it for chunking.
    Supports .txt, .md, and other plain text formats.
    """
    
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            ProcessedDocument with file content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is not UTF-8
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        
        # Read with UTF-8, fallback to latin-1
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=content,
            metadata={
                "type": "text",
                "extension": file_path.suffix,
                "source": str(file_path.absolute()),
                "char_count": len(content),
            },
        )
