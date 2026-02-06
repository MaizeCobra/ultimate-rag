"""Image processor using Gemini Vision API."""
import base64
import uuid
from io import BytesIO
from pathlib import Path

import google.generativeai as genai
from PIL import Image

from src.config import get_settings
from .base import BaseProcessor, ProcessedDocument


class ImageProcessor(BaseProcessor):
    """Processor for image files using Gemini Vision.
    
    Uses Gemini's multimodal capabilities to generate detailed
    descriptions of images for semantic search.
    """
    
    # Maximum image dimension before resizing
    MAX_DIMENSION = 1024
    
    def __init__(self):
        """Initialize with Gemini Vision model."""
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_flash_model)
    
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ProcessedDocument with image description
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If Vision API fails
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        
        # Load and optionally resize image
        image_bytes, dimensions = self._prepare_image(file_path)
        
        # Get MIME type
        suffix = file_path.suffix.lower()
        mime_type = self._get_mime_type(suffix)
        
        # Generate description using Vision API
        description = self._describe_image(image_bytes, mime_type)
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=description,
            metadata={
                "type": "image",
                "format": suffix[1:],
                "source": str(file_path.absolute()),
                "width": dimensions[0],
                "height": dimensions[1],
            },
        )
    
    def _prepare_image(self, file_path: Path) -> tuple[bytes, tuple[int, int]]:
        """Load and resize image if needed.
        
        Args:
            file_path: Path to image
            
        Returns:
            Tuple of (image_bytes, (width, height))
        """
        with Image.open(file_path) as img:
            original_size = img.size
            
            # Resize if too large
            if max(img.size) > self.MAX_DIMENSION:
                ratio = self.MAX_DIMENSION / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed (for RGBA images)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Save to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue(), original_size
    
    def _get_mime_type(self, suffix: str) -> str:
        """Get MIME type from file extension."""
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(suffix, "image/jpeg")
    
    def _describe_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Generate description using Gemini Vision.
        
        Args:
            image_bytes: Image data
            mime_type: MIME type of image
            
        Returns:
            Detailed description of the image
        """
        prompt = """Describe this image in detail for a knowledge base. Include:
1. Main subjects and objects visible
2. Text visible in the image (if any)
3. Colors, composition, and visual style
4. Context and setting
5. Any relevant technical details (charts, diagrams, etc.)

Provide a comprehensive description that would help someone search for this image."""

        try:
            response = self.model.generate_content([
                prompt,
                {"mime_type": mime_type, "data": image_bytes}
            ])
            return response.text.strip()
        except Exception as e:
            return f"[Image description unavailable: {str(e)}]"
