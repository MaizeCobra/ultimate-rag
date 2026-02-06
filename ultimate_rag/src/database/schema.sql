-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
-- Vector store table (matches n8n's documents_pg)
CREATE TABLE IF NOT EXISTS documents_pg (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(3072) -- Gemini embedding dimension
);
-- Create HNSW index for fast similarity search
-- NOTE: 3072-dim vectors exceed 8KB page size for HNSW. Indexing requires quantization or larger pages.
-- CREATE INDEX IF NOT EXISTS documents_pg_embedding_idx ON documents_pg USING hnsw (embedding vector_cosine_ops);
-- Document metadata table
CREATE TABLE IF NOT EXISTS document_metadata (
    id TEXT PRIMARY KEY,
    filename TEXT,
    file_type TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    url TEXT,
    schema TEXT -- Column schema for tabular files
);
-- Tabular data storage (rows from CSV/XLSX)
CREATE TABLE IF NOT EXISTS document_rows (
    id SERIAL PRIMARY KEY,
    dataset_id TEXT REFERENCES document_metadata(id) ON DELETE CASCADE,
    row_data JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS document_rows_dataset_idx ON document_rows(dataset_id);
-- Chat memory table
CREATE TABLE IF NOT EXISTS chat_memory (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('human', 'ai')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chat_memory_session_idx ON chat_memory(session_id);