"""Embedding generation for the ingestion pipeline.

Uses Google's Gemini embedding model to generate vector embeddings
for text chunks, enabling semantic search and retrieval.
"""
import asyncio
import time
from typing import Sequence

import google.generativeai as genai

from src.config import get_settings


class Embedder:
    """Generates embeddings using Gemini embedding model.
    
    Attributes:
        model: The embedding model name (default: embedding-001)
        rate_limit_delay: Seconds to wait between API calls (default: 0.1)
    """
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """Initialize the embedder with Gemini API.
        
        Args:
            rate_limit_delay: Delay between API calls for rate limiting
        """
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = settings.gemini_embedding_model
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0.0
    
    def _wait_for_rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call_time = time.time()
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            768-dimensional embedding vector
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)
    
    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding (used internally)."""
        self._wait_for_rate_limit()
        
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e
    
    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Note:
            Currently processes sequentially with rate limiting.
            Future versions may use batch API when available.
        """
        embeddings = []
        for text in texts:
            if text and text.strip():
                embedding = await self.embed(text)
                embeddings.append(embedding)
            else:
                # Return zero vector for empty texts
                embeddings.append([0.0] * 768)
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings (3072 for gemini-embedding-001)."""
        return 3072
