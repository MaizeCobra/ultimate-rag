"""File type detection for the ingestion pipeline.

Detects file categories based on MIME type and extension
to route files to the appropriate processor.
"""
import mimetypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileCategory(Enum):
    """Categories of files supported by the ingestion pipeline."""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    UNKNOWN = "unknown"


# MIME type to category mapping
MIME_MAPPING: dict[str, FileCategory] = {
    # Text files
    "text/plain": FileCategory.TEXT,
    "text/markdown": FileCategory.TEXT,
    
    # JSON
    "application/json": FileCategory.JSON,
    
    # Spreadsheets
    "text/csv": FileCategory.CSV,
    "application/csv": FileCategory.CSV,
    "application/vnd.ms-excel": FileCategory.XLSX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileCategory.XLSX,
    
    # Images
    "image/jpeg": FileCategory.IMAGE,
    "image/jpg": FileCategory.IMAGE,
    "image/png": FileCategory.IMAGE,
    "image/gif": FileCategory.IMAGE,
    "image/webp": FileCategory.IMAGE,
    "image/bmp": FileCategory.IMAGE,
    
    # Audio
    "audio/mpeg": FileCategory.AUDIO,
    "audio/mp3": FileCategory.AUDIO,
    "audio/wav": FileCategory.AUDIO,
    "audio/x-wav": FileCategory.AUDIO,
    "audio/ogg": FileCategory.AUDIO,
    "audio/flac": FileCategory.AUDIO,
    "audio/aac": FileCategory.AUDIO,
    
    # Video
    "video/mp4": FileCategory.VIDEO,
    "video/mpeg": FileCategory.VIDEO,
    "video/webm": FileCategory.VIDEO,
    "video/quicktime": FileCategory.VIDEO,
    "video/x-msvideo": FileCategory.VIDEO,
    
    # PDF
    "application/pdf": FileCategory.PDF,
}

# Extension fallback for when MIME detection fails
EXTENSION_MAPPING: dict[str, FileCategory] = {
    ".txt": FileCategory.TEXT,
    ".md": FileCategory.TEXT,
    ".json": FileCategory.JSON,
    ".csv": FileCategory.CSV,
    ".xlsx": FileCategory.XLSX,
    ".xls": FileCategory.XLSX,
    ".jpg": FileCategory.IMAGE,
    ".jpeg": FileCategory.IMAGE,
    ".png": FileCategory.IMAGE,
    ".gif": FileCategory.IMAGE,
    ".webp": FileCategory.IMAGE,
    ".bmp": FileCategory.IMAGE,
    ".mp3": FileCategory.AUDIO,
    ".wav": FileCategory.AUDIO,
    ".ogg": FileCategory.AUDIO,
    ".flac": FileCategory.AUDIO,
    ".aac": FileCategory.AUDIO,
    ".mp4": FileCategory.VIDEO,
    ".mpeg": FileCategory.VIDEO,
    ".webm": FileCategory.VIDEO,
    ".mov": FileCategory.VIDEO,
    ".avi": FileCategory.VIDEO,
    ".pdf": FileCategory.PDF,
}


@dataclass
class FileInfo:
    """Information about a detected file."""
    path: Path
    category: FileCategory
    mime_type: str | None
    extension: str


def detect_file_type(file_path: str | Path) -> FileInfo:
    """Detect the file type and return file information.
    
    Args:
        file_path: Path to the file (string or Path object)
        
    Returns:
        FileInfo with detected category, MIME type, and extension
        
    Example:
        >>> info = detect_file_type("document.pdf")
        >>> info.category
        <FileCategory.PDF: 'pdf'>
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    # Try MIME type detection
    mime_type, _ = mimetypes.guess_type(str(path))
    
    # Prioritize extension mapping for reliable detection
    # (Windows MIME detection can misclassify .csv as Excel)
    if extension in EXTENSION_MAPPING:
        category = EXTENSION_MAPPING[extension]
    # Fallback to MIME type if extension not recognized
    elif mime_type and mime_type in MIME_MAPPING:
        category = MIME_MAPPING[mime_type]
    else:
        category = FileCategory.UNKNOWN
    
    return FileInfo(
        path=path,
        category=category,
        mime_type=mime_type,
        extension=extension,
    )


def is_supported(file_path: str | Path) -> bool:
    """Check if a file type is supported by the pipeline.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file type is supported
    """
    info = detect_file_type(file_path)
    return info.category != FileCategory.UNKNOWN
