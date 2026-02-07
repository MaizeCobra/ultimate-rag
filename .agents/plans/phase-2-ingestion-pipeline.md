# Feature: Phase 2 - Ingestion Pipeline

The following plan should be complete, but validate documentation and codebase patterns before implementing.

## Feature Description

Build a complete ingestion pipeline that processes 8 file types (TXT, JSON, CSV, XLSX, Image, Audio, Video, PDF), chunks text content, generates embeddings, and stores everything in PostgreSQL with pgvector.

## User Story

As a **knowledge worker**
I want to **upload documents in any format**
So that **the system indexes them for AI-powered search and chat**

## Problem Statement

Currently the system has database infrastructure but no way to process and store documents. Users need to upload files of various types and have them automatically converted to searchable vector embeddings.

## Solution Statement

Create a modular ingestion pipeline with:
1. **File detector** - Routes files to correct processor by MIME type
2. **Processors** - Handle each file type (text, spreadsheet, multimodal, PDF)
3. **Chunker** - Split text into 1000-char chunks with 200-char overlap
4. **Embedder** - Generate 768-dim vectors using Gemini embedding-001
5. **Pipeline** - Orchestrate the full flow from file → database

## Feature Metadata

**Feature Type**: New Capability
**Estimated Complexity**: High
**Primary Systems Affected**: `src/ingestion/`, `src/database/`
**Dependencies**: `langchain-text-splitters`, `google-generativeai`, `pandas`, `openpyxl`, `PyMuPDF`

---

## CONTEXT REFERENCES

### Relevant Codebase Files

- [connection.py](file:///c:/Environment/RAG/ultimate_rag/src/database/connection.py) - Async pool pattern to mirror
- [config.py](file:///c:/Environment/RAG/ultimate_rag/src/config.py) (lines 21-25) - Gemini model configs
- [schema.sql](file:///c:/Environment/RAG/ultimate_rag/src/database/schema.sql) - Table structures
- [test_database.py](file:///c:/Environment/RAG/ultimate_rag/tests/test_database.py) - pytest-asyncio patterns
- [extractor.py](file:///c:/Environment/RAG/Lumosity-RAG/extractor.py) - `PDFExtractor` class to integrate
- [image_handler.py](file:///c:/Environment/RAG/Lumosity-RAG/image_handler.py) - `ImageHandler` for PDF images
- [vision_describer.py](file:///c:/Environment/RAG/Lumosity-RAG/vision_describer.py) - Gemini Vision API pattern

### New Files to Create

```
src/ingestion/
├── file_detector.py       # MIME type → category mapping
├── chunker.py             # RecursiveCharacterTextSplitter wrapper
├── embedder.py            # Gemini embedding-001 wrapper
├── pipeline.py            # Orchestrator: detect → process → chunk → embed → store
└── processors/
    ├── base.py            # BaseProcessor ABC
    ├── text.py            # TXT, JSON processor
    ├── spreadsheet.py     # CSV, XLSX → document_rows
    ├── image.py           # Gemini vision (gemini-2.5-flash)
    ├── audio.py           # Gemini transcription (gemini-2.5-flash)
    ├── video.py           # Gemini analysis (gemini-2.5-pro)
    └── pdf.py             # Lumosity-RAG integration
tests/
├── test_ingestion.py      # Unit tests for processors
└── conftest.py            # Shared fixtures
```

### Relevant Documentation

- [LangChain Text Splitters](https://docs.langchain.com/oss/python/langchain/rag) - `RecursiveCharacterTextSplitter`
- [Gemini File API](https://ai.google.dev/api/files) - Audio/video upload and analysis
- [Gemini Embeddings](https://ai.google.dev/gemini-api/docs/embeddings) - `embed_content()` method

### Patterns to Follow

**Async Context Manager** (from connection.py):
```python
@asynccontextmanager
async def get_connection() -> AsyncGenerator[Connection, None]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn
```

**Settings Access** (from config.py):
```python
from src.config import get_settings
settings = get_settings()
model = settings.gemini_flash_model
```

**Gemini Vision** (from vision_describer.py):
```python
genai.configure(api_key=self.api_key)
model = genai.GenerativeModel(Config.VISION_MODEL)
response = model.generate_content([prompt, {"mime_type": "image/png", "data": image_bytes}])
```

**Test Pattern** (from test_database.py):
```python
@pytest_asyncio.fixture(autouse=True)
async def setup_teardown():
    yield
    await close_pool()

@pytest.mark.asyncio
async def test_something():
    async with get_connection() as conn:
        ...
```

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation (Tasks 1-4)

Create base classes and file detection infrastructure.

### Phase 2: Text Processing (Tasks 5-8)

Implement text/JSON and spreadsheet processors, plus chunking.

### Phase 3: Multimodal (Tasks 9-14)

Image, audio, video processors using Gemini API.

### Phase 4: PDF & Pipeline (Tasks 15-19)

PDF processor with Lumosity-RAG integration and full pipeline.

### Phase 5: Testing (Tasks 20-22)

Comprehensive test suite for all processors.

---

## STEP-BY-STEP TASKS

### Task 1: CREATE `src/ingestion/file_detector.py`

- **IMPLEMENT**: `FileCategory` enum, `FileInfo` dataclass, `detect_file_type()` function
- **PATTERN**: Use stdlib `mimetypes` + extension fallback
- **IMPORTS**: `mimetypes`, `pathlib`, `enum`, `dataclasses`
- **VALIDATE**: `python -c "from src.ingestion.file_detector import detect_file_type; print(detect_file_type('test.pdf'))"`

```python
# Key structure:
class FileCategory(Enum):
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    UNKNOWN = "unknown"

MIME_MAPPING = {
    "text/plain": FileCategory.TEXT,
    "application/json": FileCategory.JSON,
    "text/csv": FileCategory.CSV,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileCategory.XLSX,
    "image/jpeg": FileCategory.IMAGE,
    "image/png": FileCategory.IMAGE,
    "audio/mpeg": FileCategory.AUDIO,
    "audio/wav": FileCategory.AUDIO,
    "video/mp4": FileCategory.VIDEO,
    "application/pdf": FileCategory.PDF,
}
```

---

### Task 2: CREATE `src/ingestion/processors/base.py`

- **IMPLEMENT**: `BaseProcessor` ABC with `process()` method returning `ProcessedDocument`
- **PATTERN**: Abstract base class with dataclass output
- **IMPORTS**: `abc`, `dataclasses`, `pathlib`

```python
@dataclass
class ProcessedDocument:
    file_id: str
    title: str
    content: str  # Text to chunk/embed
    metadata: dict
    rows: list[dict] | None = None  # For tabular data
```

---

### Task 3: CREATE `src/ingestion/chunker.py`

- **IMPLEMENT**: `TextChunker` class wrapping `RecursiveCharacterTextSplitter`
- **PATTERN**: 1000 chars, 200 overlap, track start indices
- **IMPORTS**: `langchain_text_splitters`
- **GOTCHA**: Use `split_text()` for raw strings, not `split_documents()`
- **VALIDATE**: `python -c "from src.ingestion.chunker import TextChunker; c = TextChunker(); print(len(c.chunk('x'*2500)))"`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    
    def chunk(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
```

---

### Task 4: CREATE `src/ingestion/embedder.py`

- **IMPLEMENT**: `Embedder` class with async `embed()` and `embed_batch()` methods
- **PATTERN**: Use `genai.embed_content()` with task_type="retrieval_document"
- **IMPORTS**: `google.generativeai as genai`
- **GOTCHA**: Gemini rate limits - add delay between batches
- **VALIDATE**: `python -c "from src.ingestion.embedder import Embedder; import asyncio; e = Embedder(); print(len(asyncio.run(e.embed('test'))))"`

```python
import google.generativeai as genai
from src.config import get_settings

class Embedder:
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = settings.gemini_embedding_model
    
    async def embed(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]  # 768-dim vector
```

---

### Task 5: CREATE `src/ingestion/processors/text.py`

- **IMPLEMENT**: `TextProcessor` for TXT files - read content directly
- **PATTERN**: Inherit from `BaseProcessor`, use `pathlib.Path.read_text()`
- **IMPORTS**: `pathlib`, `uuid`
- **VALIDATE**: `python -c "from src.ingestion.processors.text import TextProcessor; p = TextProcessor(); print(p.process('README.md'))"`

---

### Task 6: CREATE `src/ingestion/processors/json_processor.py`

- **IMPLEMENT**: `JsonProcessor` - stringify JSON for RAG
- **PATTERN**: `json.dumps(..., indent=2)` for readable embedding
- **IMPORTS**: `json`, `pathlib`
- **GOTCHA**: Handle nested structures, arrays

---

### Task 7: CREATE `src/ingestion/processors/spreadsheet.py`

- **IMPLEMENT**: `SpreadsheetProcessor` for CSV/XLSX → JSONB rows + summary text
- **PATTERN**: Use pandas, store each row in `document_rows` table
- **IMPORTS**: `pandas`, `openpyxl` (for xlsx)
- **GOTCHA**: Schema extraction for SQL tool later

```python
import pandas as pd

class SpreadsheetProcessor(BaseProcessor):
    def process(self, file_path: Path) -> ProcessedDocument:
        df = pd.read_csv(file_path) if file_path.suffix == ".csv" else pd.read_excel(file_path)
        
        # Create summary text for embedding
        schema = ", ".join(df.columns.tolist())
        summary = f"Dataset with columns: {schema}. Contains {len(df)} rows."
        
        # Convert rows to dicts for document_rows table
        rows = df.to_dict(orient="records")
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=summary,
            metadata={"schema": schema, "row_count": len(df)},
            rows=rows,
        )
```

---

### Task 8: UPDATE `src/ingestion/__init__.py`

- **IMPLEMENT**: Export all processors and utilities
- **VALIDATE**: `python -c "from src.ingestion import TextChunker, Embedder, detect_file_type"`

---

### Task 9: CREATE `src/ingestion/processors/image.py`

- **IMPLEMENT**: `ImageProcessor` using Gemini Vision (gemini-2.5-flash)
- **PATTERN**: Mirror `VisionDescriber.describe_image()` from Lumosity-RAG
- **IMPORTS**: `google.generativeai`, `PIL.Image`
- **GOTCHA**: Resize large images before API call

```python
class ImageProcessor(BaseProcessor):
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_flash_model)
    
    def process(self, file_path: Path) -> ProcessedDocument:
        image_bytes = file_path.read_bytes()
        response = self.model.generate_content([
            "Describe this image in detail for a knowledge base.",
            {"mime_type": f"image/{file_path.suffix[1:]}", "data": image_bytes}
        ])
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content=response.text,
            metadata={"type": "image", "source": str(file_path)},
        )
```

---

### Task 10: CREATE `src/ingestion/processors/audio.py`

- **IMPLEMENT**: `AudioProcessor` using Gemini File API + gemini-2.5-flash
- **PATTERN**: Upload file first, then generate content
- **IMPORTS**: `google.genai` (new SDK)
- **GOTCHA**: Wait for file processing before generating

```python
from google import genai

class AudioProcessor(BaseProcessor):
    def __init__(self):
        settings = get_settings()
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_flash_model
    
    def process(self, file_path: Path) -> ProcessedDocument:
        # Upload audio file
        uploaded = self.client.files.upload(file=file_path)
        
        # Generate transcription/description
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded, "Transcribe and summarize this audio."]
        )
        return ProcessedDocument(...)
```

---

### Task 11: CREATE `src/ingestion/processors/video.py`

- **IMPLEMENT**: `VideoProcessor` using Gemini File API + gemini-2.5-pro
- **PATTERN**: Same as audio but use pro model for complex temporal analysis
- **GOTCHA**: Large videos may need chunking or timeout handling

---

### Task 12: CREATE `src/ingestion/processors/pdf.py`

- **IMPLEMENT**: `PDFProcessor` integrating Lumosity-RAG components
- **PATTERN**: Use `PDFExtractor`, `ImageHandler`, process 3 pages at a time
- **IMPORTS**: Copy patterns from Lumosity-RAG
- **GOTCHA**: Handle PDFs with many images efficiently

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "Lumosity-RAG"))
from extractor import PDFExtractor
from image_handler import ImageHandler

class PDFProcessor(BaseProcessor):
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_flash_model)
        self.image_handler = ImageHandler()
    
    def process(self, file_path: Path) -> ProcessedDocument:
        all_text = []
        with PDFExtractor(file_path) as extractor:
            for page in extractor.extract_all():
                all_text.append(page.text)
                # Process significant images with vision API
                for img in page.images:
                    if self.image_handler.should_process(img):
                        desc = self._describe_image(img)
                        all_text.append(f"[Image: {desc}]")
        
        return ProcessedDocument(
            file_id=str(uuid.uuid4()),
            title=file_path.name,
            content="\n".join(all_text),
            metadata={"type": "pdf", "pages": extractor.page_count},
        )
```

---

### Task 13: CREATE `src/ingestion/pipeline.py`

- **IMPLEMENT**: `IngestPipeline` orchestrator class
- **PATTERN**: detect → process → chunk → embed → store (database)
- **IMPORTS**: All processors, chunker, embedder, database connection

```python
class IngestPipeline:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.processors = {
            FileCategory.TEXT: TextProcessor(),
            FileCategory.JSON: JsonProcessor(),
            FileCategory.CSV: SpreadsheetProcessor(),
            FileCategory.XLSX: SpreadsheetProcessor(),
            FileCategory.IMAGE: ImageProcessor(),
            FileCategory.AUDIO: AudioProcessor(),
            FileCategory.VIDEO: VideoProcessor(),
            FileCategory.PDF: PDFProcessor(),
        }
    
    async def ingest(self, file_path: Path) -> str:
        """Ingest a file and return its document ID."""
        # 1. Detect file type
        file_info = detect_file_type(file_path)
        
        # 2. Process with appropriate processor
        processor = self.processors[file_info.category]
        doc = processor.process(file_path)
        
        # 3. Store metadata
        await self._store_metadata(doc)
        
        # 4. Store rows if tabular
        if doc.rows:
            await self._store_rows(doc.file_id, doc.rows)
        
        # 5. Chunk content
        chunks = self.chunker.chunk(doc.content)
        
        # 6. Embed and store each chunk
        for chunk in chunks:
            embedding = await self.embedder.embed(chunk)
            await self._store_document(doc.file_id, chunk, embedding, doc.metadata)
        
        return doc.file_id
```

---

### Task 14: UPDATE `src/database/__init__.py`

- **IMPLEMENT**: Add functions for storing documents: `store_document`, `store_metadata`, `store_rows`
- **VALIDATE**: `python -c "from src.database import store_document"`

---

### Task 15: CREATE `tests/conftest.py`

- **IMPLEMENT**: Shared fixtures for tests (sample files, database setup)
- **PATTERN**: Mirror test_database.py setup_teardown fixture

---

### Task 16: CREATE `tests/test_file_detector.py`

- **IMPLEMENT**: Tests for all 8 file type detections
- **VALIDATE**: `pytest tests/test_file_detector.py -v`

---

### Task 17: CREATE `tests/test_chunker.py`

- **IMPLEMENT**: Tests for chunking edge cases (short text, exact boundary, long text)
- **VALIDATE**: `pytest tests/test_chunker.py -v`

---

### Task 18: CREATE `tests/test_processors.py`

- **IMPLEMENT**: Unit tests for each processor with sample files
- **GOTCHA**: Mock Gemini API calls for image/audio/video tests
- **VALIDATE**: `pytest tests/test_processors.py -v`

---

### Task 19: CREATE `tests/test_pipeline.py`

- **IMPLEMENT**: Integration test for full pipeline flow
- **PATTERN**: Create temp file → ingest → verify in database
- **VALIDATE**: `pytest tests/test_pipeline.py -v`

---

### Task 20: UPDATE `src/ingestion/__init__.py`

- **IMPLEMENT**: Clean exports for public API
- **VALIDATE**: `python -c "from src.ingestion import IngestPipeline"`

---

## TESTING STRATEGY

### Unit Tests

- `test_file_detector.py` - All MIME type mappings
- `test_chunker.py` - Chunk sizes and overlap
- `test_processors.py` - Each processor with real sample files (mock API for multimodal)

### Integration Tests

- `test_pipeline.py` - Full flow from file to database

### Edge Cases

- Empty files
- Very large files (>100MB)
- Corrupt PDF
- Unsupported file types
- Files with no extension

---

## VALIDATION COMMANDS

### Level 1: Syntax & Style

```bash
python -m py_compile src/ingestion/*.py
python -m py_compile src/ingestion/processors/*.py
```

### Level 2: Unit Tests

```bash
pytest tests/test_file_detector.py -v
pytest tests/test_chunker.py -v
pytest tests/test_processors.py -v
```

### Level 3: Integration Tests

```bash
pytest tests/test_pipeline.py -v
pytest tests/ -v  # Full suite
```

### Level 4: Manual Validation

```bash
# Test with real files
python -c "
from src.ingestion import IngestPipeline
import asyncio

async def test():
    pipeline = IngestPipeline()
    doc_id = await pipeline.ingest('test_files/sample.txt')
    print(f'Ingested: {doc_id}')

asyncio.run(test())
"
```

---

## ACCEPTANCE CRITERIA

- [ ] All 8 file types processed correctly
- [ ] Chunking produces consistent 1000-char segments with 200-char overlap
- [ ] Embeddings are 768-dimensional vectors
- [ ] Documents stored in `documents_pg` table
- [ ] Tabular data stored in `document_rows` table
- [ ] Metadata stored in `document_metadata` table
- [ ] All tests pass (target: 80%+ coverage)
- [ ] Pipeline handles errors gracefully

---

## NOTES

### Lumosity-RAG Integration

The existing Lumosity-RAG library in the parent directory provides battle-tested PDF extraction. Rather than rewriting, we import its components directly:
- `PDFExtractor` - PyMuPDF-based text/image extraction
- `ImageHandler` - Image filtering and resizing
- `VisionDescriber` - Gemini Vision API wrapper (reference pattern only)

### Rate Limiting

Gemini API has rate limits. Implement exponential backoff in embedder and add delays between multimodal API calls (0.5s minimum).

### Batch Processing

For large files, consider:
1. Streaming chunks rather than loading entire file
2. Batch embedding requests (up to 100 texts per call)
3. Progress callbacks for UI feedback

### Future Extensions

- Add OCR fallback for scanned PDFs
- Support additional formats (DOCX, PPTX)
- Add file deduplication by content hash
