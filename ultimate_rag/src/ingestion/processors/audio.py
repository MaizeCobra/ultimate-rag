"""Audio processor using Gemini File API."""
import time
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from src.config import get_settings
from .base import BaseProcessor, ProcessedDocument


class AudioProcessor(BaseProcessor):
    """Processor for audio files using Gemini.
    
    Uploads audio to Gemini File API and generates
    transcription/summary for semantic search.
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
    
    def __init__(self):
        """Initialize with Gemini client."""
        settings = get_settings()
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_flash_model
    
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            ProcessedDocument with transcription/summary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {suffix}")
        
        # Upload file to Gemini
        uploaded_file = self._upload_file(file_path)
        
        # Generate transcription/summary
        content = self._process_audio(uploaded_file)
        
        # Get file size for metadata
        file_size = file_path.stat().st_size
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=content,
            metadata={
                "type": "audio",
                "format": suffix[1:],
                "source": str(file_path.absolute()),
                "file_size_bytes": file_size,
            },
        )
    
    def _upload_file(self, file_path: Path) -> types.File:
        """Upload file to Gemini File API.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Uploaded file reference
        """
        uploaded = self.client.files.upload(file=file_path)
        
        # Wait for file to be processed
        while uploaded.state == "PROCESSING":
            time.sleep(1)
            uploaded = self.client.files.get(name=uploaded.name)
        
        if uploaded.state == "FAILED":
            raise RuntimeError(f"File upload failed: {uploaded.name}")
        
        return uploaded
    
    def _process_audio(self, uploaded_file: types.File) -> str:
        """Generate transcription and summary.
        
        Args:
            uploaded_file: Uploaded file reference
            
        Returns:
            Transcription and summary text
        """
        prompt = """Please process this audio file and provide:

1. **Full Transcription**: Transcribe all spoken content verbatim.

2. **Summary**: A brief summary of the main topics discussed.

3. **Key Points**: List the main points or takeaways.

4. **Speakers**: If multiple speakers, identify them as Speaker 1, Speaker 2, etc.

Format the output clearly with sections."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[uploaded_file, prompt],
            )
            return response.text.strip()
        except Exception as e:
            return f"[Audio processing failed: {str(e)}]"
