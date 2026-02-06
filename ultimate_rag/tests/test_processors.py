"""Tests for file processors."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ingestion.processors import (
    TextProcessor,
    JsonProcessor,
    SpreadsheetProcessor,
    ProcessedDocument,
)


class TestTextProcessor:
    """Test text file processor."""
    
    def test_process_text_file(self, sample_text_file):
        """Test processing a text file."""
        processor = TextProcessor()
        doc = processor.process(sample_text_file)
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.title == sample_text_file.name
        assert len(doc.content) > 0
        assert doc.metadata["type"] == "text"
    
    def test_process_missing_file(self, tmp_path):
        """Test processing non-existent file raises error."""
        processor = TextProcessor()
        fake_file = tmp_path / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            processor.process(fake_file)


class TestJsonProcessor:
    """Test JSON file processor."""
    
    def test_process_json_file(self, sample_json_file):
        """Test processing a JSON file."""
        processor = JsonProcessor()
        doc = processor.process(sample_json_file)
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.title == sample_json_file.name
        assert "Test Document" in doc.content
        assert doc.metadata["type"] == "json"
    
    def test_process_nested_json(self, tmp_path):
        """Test processing nested JSON."""
        processor = JsonProcessor()
        file = tmp_path / "nested.json"
        data = {
            "level1": {
                "level2": {
                    "level3": "deep value"
                }
            }
        }
        file.write_text(json.dumps(data))
        
        doc = processor.process(file)
        assert "level1" in doc.content
        assert "deep value" in doc.content
    
    def test_process_json_array(self, tmp_path):
        """Test processing JSON array."""
        processor = JsonProcessor()
        file = tmp_path / "array.json"
        data = [{"name": "item1"}, {"name": "item2"}]
        file.write_text(json.dumps(data))
        
        doc = processor.process(file)
        assert "item1" in doc.content
        assert "item2" in doc.content


class TestSpreadsheetProcessor:
    """Test spreadsheet processor."""
    
    def test_process_csv_file(self, sample_csv_file):
        """Test processing a CSV file."""
        processor = SpreadsheetProcessor()
        doc = processor.process(sample_csv_file)
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.title == sample_csv_file.name
        assert doc.metadata["type"] == "spreadsheet"
        assert doc.metadata["row_count"] == 3
        assert doc.rows is not None
        assert len(doc.rows) == 3
    
    def test_process_xlsx_file(self, sample_xlsx_file):
        """Test processing an XLSX file."""
        processor = SpreadsheetProcessor()
        doc = processor.process(sample_xlsx_file)
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.metadata["format"] == "xlsx"
        assert doc.rows is not None
    
    def test_csv_summary_contains_schema(self, sample_csv_file):
        """Test that CSV summary includes column names."""
        processor = SpreadsheetProcessor()
        doc = processor.process(sample_csv_file)
        
        assert "id" in doc.content
        assert "name" in doc.content
        assert "value" in doc.content
        assert "category" in doc.content
    
    def test_csv_rows_are_dicts(self, sample_csv_file):
        """Test that rows are dictionaries."""
        processor = SpreadsheetProcessor()
        doc = processor.process(sample_csv_file)
        
        assert all(isinstance(row, dict) for row in doc.rows)
        assert doc.rows[0]["name"] == "Item A"


class TestImageProcessor:
    """Test image processor with mocked Gemini API."""
    
    @patch("src.ingestion.processors.image.genai")
    def test_process_image_mocked(self, mock_genai, image_fixtures):
        """Test image processing with mocked API."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = "A sample image description"
        mock_genai.GenerativeModel.return_value = mock_model
        
        from src.ingestion.processors.image import ImageProcessor
        processor = ImageProcessor()
        doc = processor.process(image_fixtures[0])
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.metadata["type"] == "image"


class TestAudioProcessor:
    """Test audio processor with mocked Gemini API."""
    
    @patch("src.ingestion.processors.audio.genai")
    def test_audio_processor_init(self, mock_genai):
        """Test audio processor initialization."""
        from src.ingestion.processors.audio import AudioProcessor
        processor = AudioProcessor()
        assert processor is not None


class TestVideoProcessor:
    """Test video processor with mocked Gemini API."""
    
    @patch("src.ingestion.processors.video.genai")
    def test_video_processor_init(self, mock_genai):
        """Test video processor initialization."""
        from src.ingestion.processors.video import VideoProcessor
        processor = VideoProcessor()
        assert processor is not None
