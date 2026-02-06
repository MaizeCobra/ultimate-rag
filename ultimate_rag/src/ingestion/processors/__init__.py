"""File type processors for ingestion pipeline."""
from .base import BaseProcessor, ProcessedDocument
from .text import TextProcessor
from .json_processor import JsonProcessor
from .spreadsheet import SpreadsheetProcessor
from .image import ImageProcessor
from .audio import AudioProcessor
from .video import VideoProcessor
from .pdf import PDFProcessor

__all__ = [
    "BaseProcessor",
    "ProcessedDocument",
    "TextProcessor",
    "JsonProcessor",
    "SpreadsheetProcessor",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "PDFProcessor",
]
