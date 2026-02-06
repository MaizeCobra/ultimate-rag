"""Shared pytest fixtures for ingestion tests."""
import json
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from src.database import close_pool, init_schema


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest_asyncio.fixture(autouse=True)
async def setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Initialize database schema
    await init_schema()
    yield
    # Teardown: Close database connections
    await close_pool()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_text_file(tmp_path) -> Path:
    """Create a sample text file."""
    file = tmp_path / "sample.txt"
    file.write_text("This is a test document for the RAG pipeline.\n" * 50)
    return file


@pytest.fixture
def sample_json_file(tmp_path) -> Path:
    """Create a sample JSON file."""
    file = tmp_path / "sample.json"
    data = {
        "title": "Test Document",
        "content": "This is test content",
        "tags": ["test", "sample", "json"],
        "metadata": {
            "author": "Test User",
            "version": 1
        }
    }
    file.write_text(json.dumps(data, indent=2))
    return file


@pytest.fixture
def sample_csv_file(tmp_path) -> Path:
    """Create a sample CSV file."""
    file = tmp_path / "sample.csv"
    file.write_text(
        "id,name,value,category\n"
        "1,Item A,100.50,Electronics\n"
        "2,Item B,25.99,Books\n"
        "3,Item C,50.00,Clothing\n"
    )
    return file


@pytest.fixture
def sample_xlsx_file(tmp_path) -> Path:
    """Create a sample XLSX file using pandas."""
    try:
        import pandas as pd
        file = tmp_path / "sample.xlsx"
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Item A", "Item B", "Item C"],
            "value": [100.50, 25.99, 50.00],
            "category": ["Electronics", "Books", "Clothing"]
        })
        df.to_excel(file, index=False)
        return file
    except ImportError:
        pytest.skip("pandas or openpyxl not installed")


@pytest.fixture
def audio_fixture() -> Path:
    """Return path to sample audio file if it exists."""
    audio_path = FIXTURES_DIR / "sample_audio.mp3"
    if not audio_path.exists():
        pytest.skip("Sample audio file not found in fixtures")
    return audio_path


@pytest.fixture
def video_fixture() -> Path:
    """Return path to sample video file if it exists."""
    video_path = FIXTURES_DIR / "sample_video.mp4"
    if not video_path.exists():
        pytest.skip("Sample video file not found in fixtures")
    return video_path


@pytest.fixture
def image_fixtures() -> list[Path]:
    """Return paths to sample image files."""
    images = list(FIXTURES_DIR.glob("*.png"))
    if not images:
        pytest.skip("No sample images found in fixtures")
    return images
