"""JSON file processor."""
import json
import uuid
from pathlib import Path
from typing import Any

from .base import BaseProcessor, ProcessedDocument


class JsonProcessor(BaseProcessor):
    """Processor for JSON files.
    
    Converts JSON to a readable string format for embedding,
    which works well for semantic search over structured data.
    """
    
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ProcessedDocument with readable JSON content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        
        # Load JSON data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert to readable string for embedding
        content = self._json_to_text(data)
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=content,
            metadata={
                "type": "json",
                "source": str(file_path.absolute()),
                "structure": self._describe_structure(data),
            },
        )
    
    def _json_to_text(self, data: Any, prefix: str = "") -> str:
        """Convert JSON data to readable text.
        
        Args:
            data: JSON data (dict, list, or primitive)
            prefix: Key prefix for nested structures
            
        Returns:
            Human-readable text representation
        """
        lines = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.append(f"{full_key}:")
                    lines.append(self._json_to_text(value, full_key))
                else:
                    lines.append(f"{full_key}: {value}")
        elif isinstance(data, list):
            if not data:
                lines.append(f"{prefix}: (empty list)")
            elif all(isinstance(item, (str, int, float, bool, type(None))) for item in data):
                # Simple list
                lines.append(f"{prefix}: {', '.join(str(item) for item in data)}")
            else:
                # List of objects
                for i, item in enumerate(data):
                    lines.append(f"{prefix}[{i}]:")
                    lines.append(self._json_to_text(item, f"{prefix}[{i}]"))
        else:
            lines.append(f"{prefix}: {data}")
        
        return "\n".join(lines)
    
    def _describe_structure(self, data: Any) -> str:
        """Describe the JSON structure.
        
        Args:
            data: JSON data
            
        Returns:
            String describing the structure (e.g., "object with 5 keys")
        """
        if isinstance(data, dict):
            return f"object with {len(data)} keys"
        elif isinstance(data, list):
            return f"array with {len(data)} items"
        else:
            return type(data).__name__
