"""Live integration tests with real Gemini API calls.

These tests use actual API calls to verify the full pipeline works.
Run with: pytest tests/test_live_api.py -v -s
"""
import pytest
import pytest_asyncio
from pathlib import Path

from src.ingestion import Embedder
from src.ingestion.processors.image import ImageProcessor
from src.ingestion.processors.audio import AudioProcessor
from src.ingestion.processors.video import VideoProcessor


# Path to fixtures
FIXTURES = Path(__file__).parent / "fixtures"


class TestLiveEmbedder:
    """Test Embedder with real Gemini API."""
    
    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test generating a real embedding."""
        embedder = Embedder()
        text = "This is a test document about artificial intelligence."
        
        try:
            embedding = await embedder.embed(text)
        except Exception as e:
            print(f"❌ Embedder failed with: {str(e)}")
            raise e
        
        assert isinstance(embedding, list)
        assert len(embedding) == 3072  # Gemini embedding-001 dimension
        assert all(isinstance(x, float) for x in embedding)
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}")
    
    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding generation."""
        embedder = Embedder()
        texts = [
            "First document about machine learning.",
            "Second document about data science.",
        ]
        
        embeddings = await embedder.embed_batch(texts)
        
        assert len(embeddings) == 2
        assert all(len(e) == 3072 for e in embeddings)
        print(f"✅ Batch embeddings: {len(embeddings)} texts processed")


class TestLiveImageProcessor:
    """Test ImageProcessor with real Gemini Vision API."""
    
    def test_process_image(self):
        """Test generating description for a real image."""
        # Find an image in fixtures
        images = list(FIXTURES.glob("*.png"))
        if not images:
            pytest.skip("No PNG images in fixtures")
        
        processor = ImageProcessor()
        doc = processor.process(images[0])
        
        assert doc.content  # Should have description
        assert len(doc.content) > 50  # Meaningful description
        assert doc.metadata["type"] == "image"
        
        print(f"✅ Image: {images[0].name}")
        print(f"   Description ({len(doc.content)} chars):")
        print(f"   {doc.content[:300]}...")


class TestLiveAudioProcessor:
    """Test AudioProcessor with real Gemini File API."""
    
    def test_process_audio(self):
        """Test transcribing a real audio file."""
        audio_file = FIXTURES / "sample_audio.mp3"
        if not audio_file.exists():
            pytest.skip("sample_audio.mp3 not found in fixtures")
        
        processor = AudioProcessor()
        doc = processor.process(audio_file)
        
        assert doc.content  # Should have transcription
        assert len(doc.content) > 20  # Some content
        assert doc.metadata["type"] == "audio"
        
        print(f"✅ Audio: {audio_file.name}")
        print(f"   Transcription ({len(doc.content)} chars):")
        print(f"   {doc.content[:500]}...")


class TestLiveVideoProcessor:
    """Test VideoProcessor with real Gemini File API."""
    
    def test_process_video(self):
        """Test analyzing a real video file."""
        video_file = FIXTURES / "sample_video.mp4"
        if not video_file.exists():
            pytest.skip("sample_video.mp4 not found in fixtures")
        
        processor = VideoProcessor()
        doc = processor.process(video_file)
        
        assert doc.content  # Should have analysis
        assert len(doc.content) > 50  # Meaningful analysis
        assert doc.metadata["type"] == "video"
        
        print(f"✅ Video: {video_file.name}")
        print(f"   Analysis ({len(doc.content)} chars):")
        print(f"   {doc.content[:500]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
