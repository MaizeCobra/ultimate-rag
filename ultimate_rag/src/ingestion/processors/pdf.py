"""PDF processor integrating Lumosity-RAG components."""
import sys
import uuid
from pathlib import Path

import google.generativeai as genai

from src.config import get_settings
from .base import BaseProcessor, ProcessedDocument

# Add Lumosity-RAG to path for imports
LUMOSITY_RAG_PATH = Path(__file__).parent.parent.parent.parent.parent / "Lumosity-RAG"
if LUMOSITY_RAG_PATH.exists():
    sys.path.insert(0, str(LUMOSITY_RAG_PATH))


class PDFProcessor(BaseProcessor):
    """Processor for PDF files using Lumosity-RAG.
    
    Integrates with the existing Lumosity-RAG library for
    text extraction and image processing from PDFs.
    """
    
    def __init__(self):
        """Initialize with Gemini model for image descriptions."""
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_flash_model)
        
        # Initialize Lumosity-RAG components if available
        self._extractor_class = None
        self._image_handler = None
        self._load_lumosity_components()
    
    def _load_lumosity_components(self) -> None:
        """Load Lumosity-RAG components if available."""
        try:
            from extractor import PDFExtractor
            from image_handler import ImageHandler
            
            self._extractor_class = PDFExtractor
            self._image_handler = ImageHandler()
        except ImportError:
            # Lumosity-RAG not available, use fallback
            pass
    
    def process(self, file_path: Path) -> ProcessedDocument:
        """Process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with extracted text and image descriptions
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        self.validate_file(file_path)
        
        if self._extractor_class is not None:
            return self._process_with_lumosity(file_path)
        else:
            return self._process_fallback(file_path)
    
    def _process_with_lumosity(self, file_path: Path) -> ProcessedDocument:
        """Process PDF using Lumosity-RAG extractor.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            ProcessedDocument with full extraction
        """
        all_text = []
        page_count = 0
        image_count = 0
        
        with self._extractor_class(file_path) as extractor:
            page_count = extractor.page_count
            
            for page in extractor.extract_all():
                # Add page text
                if page.text.strip():
                    all_text.append(f"--- Page {page.page_num} ---")
                    all_text.append(page.text)
                
                # Process images
                for img in page.images:
                    if self._image_handler and self._image_handler.should_process(img):
                        desc = self._describe_image(img)
                        all_text.append(f"[Image on page {page.page_num}: {desc}]")
                        image_count += 1
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content="\n".join(all_text),
            metadata={
                "type": "pdf",
                "source": str(file_path.absolute()),
                "page_count": page_count,
                "image_count": image_count,
            },
        )
    
    def _process_fallback(self, file_path: Path) -> ProcessedDocument:
        """Fallback PDF processing using PyMuPDF directly.
        
        Args:
            file_path: Path to PDF
            
        Returns:
            ProcessedDocument with text extraction
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
        
        all_text = []
        page_count = 0
        
        with fitz.open(file_path) as doc:
            page_count = len(doc)
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    all_text.append(f"--- Page {page_num} ---")
                    all_text.append(text)
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content="\n".join(all_text),
            metadata={
                "type": "pdf",
                "source": str(file_path.absolute()),
                "page_count": page_count,
            },
        )
    
    def _describe_image(self, img) -> str:
        """Describe an image using Gemini Vision.
        
        Args:
            img: ExtractedImage from Lumosity-RAG
            
        Returns:
            Description string
        """
        try:
            # Resize for API if needed
            image_bytes = img.image_bytes
            if self._image_handler:
                image_bytes = self._image_handler.resize_for_api(image_bytes)
            
            response = self.model.generate_content([
                "Describe this image briefly for a document knowledge base.",
                {"mime_type": "image/png", "data": image_bytes}
            ])
            return response.text.strip()
        except Exception as e:
            return f"[Image description unavailable: {str(e)}]"
