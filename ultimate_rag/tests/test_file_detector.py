"""Tests for file type detection."""
import pytest
from pathlib import Path

from src.ingestion import FileCategory, detect_file_type, is_supported


class TestFileDetector:
    """Test file type detection."""
    
    def test_detect_text_file(self):
        """Test TXT file detection."""
        info = detect_file_type("document.txt")
        assert info.category == FileCategory.TEXT
        assert info.extension == ".txt"
    
    def test_detect_markdown_file(self):
        """Test Markdown file detection."""
        info = detect_file_type("readme.md")
        assert info.category == FileCategory.TEXT
        assert info.extension == ".md"
    
    def test_detect_json_file(self):
        """Test JSON file detection."""
        info = detect_file_type("data.json")
        assert info.category == FileCategory.JSON
        assert info.extension == ".json"
    
    def test_detect_csv_file(self):
        """Test CSV file detection."""
        info = detect_file_type("data.csv")
        assert info.category == FileCategory.CSV
        assert info.extension == ".csv"
    
    def test_detect_xlsx_file(self):
        """Test XLSX file detection."""
        info = detect_file_type("spreadsheet.xlsx")
        assert info.category == FileCategory.XLSX
        assert info.extension == ".xlsx"
    
    def test_detect_xls_file(self):
        """Test legacy XLS file detection."""
        info = detect_file_type("legacy.xls")
        assert info.category == FileCategory.XLSX
        assert info.extension == ".xls"
    
    def test_detect_jpeg_image(self):
        """Test JPEG image detection."""
        info = detect_file_type("photo.jpg")
        assert info.category == FileCategory.IMAGE
        assert info.extension == ".jpg"
    
    def test_detect_png_image(self):
        """Test PNG image detection."""
        info = detect_file_type("screenshot.png")
        assert info.category == FileCategory.IMAGE
        assert info.extension == ".png"
    
    def test_detect_mp3_audio(self):
        """Test MP3 audio detection."""
        info = detect_file_type("podcast.mp3")
        assert info.category == FileCategory.AUDIO
        assert info.extension == ".mp3"
    
    def test_detect_wav_audio(self):
        """Test WAV audio detection."""
        info = detect_file_type("recording.wav")
        assert info.category == FileCategory.AUDIO
        assert info.extension == ".wav"
    
    def test_detect_mp4_video(self):
        """Test MP4 video detection."""
        info = detect_file_type("video.mp4")
        assert info.category == FileCategory.VIDEO
        assert info.extension == ".mp4"
    
    def test_detect_pdf_file(self):
        """Test PDF file detection."""
        info = detect_file_type("document.pdf")
        assert info.category == FileCategory.PDF
        assert info.extension == ".pdf"
    
    def test_detect_unknown_file(self):
        """Test unknown file type detection."""
        info = detect_file_type("data.xyz")
        assert info.category == FileCategory.UNKNOWN
        assert info.extension == ".xyz"
    
    def test_detect_path_object(self):
        """Test detection with Path object."""
        info = detect_file_type(Path("document.pdf"))
        assert info.category == FileCategory.PDF
    
    def test_is_supported_true(self):
        """Test is_supported for supported file."""
        assert is_supported("document.pdf") is True
        assert is_supported("data.json") is True
        assert is_supported("image.png") is True
    
    def test_is_supported_false(self):
        """Test is_supported for unsupported file."""
        assert is_supported("data.xyz") is False
        assert is_supported("binary.bin") is False
