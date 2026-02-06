"""Video processor using Gemini File API."""
import time
import uuid
from pathlib import Path

from google import genai
from google.genai import types

from src.config import get_settings
from .base import BaseProcessor, ProcessedDocument


class VideoProcessor(BaseProcessor):
    """Processor for video files using Gemini.
    
    Uploads video to Gemini File API and generates
    comprehensive analysis including visual and audio content.
    Uses the Pro model for better temporal understanding.
    """
    
    # Supported video formats
    SUPPORTED_FORMATS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".mpeg"}
    
    def __init__(self):
        """Initialize with Gemini client."""
        settings = get_settings()
        self.client = genai.Client(api_key=settings.gemini_api_key)
        # Use Pro model for complex video analysis
        self.model = settings.gemini_chat_model
    
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process a video file.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            ProcessedDocument with video analysis
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format not supported
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported video format: {suffix}")
        
        # Upload file to Gemini
        uploaded_file = self._upload_file(file_path)
        
        # Generate video analysis
        content = self._analyze_video(uploaded_file)
        
        # Get file size for metadata
        file_size = file_path.stat().st_size
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=content,
            metadata={
                "type": "video",
                "format": suffix[1:],
                "source": str(file_path.absolute()),
                "file_size_bytes": file_size,
            },
        )
    
    def _upload_file(self, file_path: Path) -> types.File:
        """Upload file to Gemini File API.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Uploaded file reference
        """
        uploaded = self.client.files.upload(file=file_path)
        
        # Wait for file to be processed (videos take longer)
        max_wait = 120  # 2 minutes max
        waited = 0
        while uploaded.state == "PROCESSING" and waited < max_wait:
            time.sleep(2)
            waited += 2
            uploaded = self.client.files.get(name=uploaded.name)
        
        if uploaded.state == "FAILED":
            raise RuntimeError(f"Video upload failed: {uploaded.name}")
        elif uploaded.state == "PROCESSING":
            raise RuntimeError("Video processing timed out")
        
        return uploaded
    
    def _analyze_video(self, uploaded_file: types.File) -> str:
        """Generate comprehensive video analysis.
        
        Args:
            uploaded_file: Uploaded file reference
            
        Returns:
            Detailed video analysis text
        """
        prompt = """Analyze this video comprehensively and provide:

1. **Overview**: Brief description of what the video is about.

2. **Visual Content**: Describe key scenes, objects, people, and visual elements.

3. **Audio/Speech**: Transcribe important dialogue or narration.

4. **Timeline**: Notable moments with approximate timestamps.

5. **Key Information**: Main topics, messages, or information conveyed.

6. **Technical Details**: Video style, quality, any on-screen text or graphics.

Provide a detailed analysis that captures all important content for searchability."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[uploaded_file, prompt],
            )
            return response.text.strip()
        except Exception as e:
            return f"[Video analysis failed: {str(e)}]"
