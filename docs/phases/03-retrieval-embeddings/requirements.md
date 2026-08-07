# Phase 3 — Retrieval & Embeddings: Requirements

## Context

Phase 2 delivered a working ingestion pipeline (`IngestPipeline.run()`) that
produces a flat `list[DocumentChunk]`. Each chunk carries:

- `content` — the text passage
- `metadata.source` — filename (e.g. `drug_manual_metformin.md`)
- `metadata.section_title` — nearest `##`/`###` header, or `None`
- `metadata.chunk_index` — zero-based position within the source doc

Phase 3 turns those chunks into a searchable knowledge base.

---

## Functional Requirements

### FR-1 Embed chunks
Given a `list[DocumentChunk]` from the Phase 2 pipeline, the system shall
convert each chunk's `content` into an embedding vector (a `list[float]`) using
AWS Bedrock's **Titan Embeddings G1 – Text** model
(`amazon.titan-embed-text-v1`) via `boto3`.

### FR-2 Store embeddings locally
Each chunk's embedding, together with its metadata (`source`,
`section_title`, `chunk_index`) and original `content`, shall be persisted to a
**local JSON file**. No database is required in this phase.

### FR-3 Search by question
Given a natural-language question string, the system shall:
1. Embed the question using the same Bedrock Titan model.
2. Compute **cosine similarity** between the question embedding and every stored
   chunk embedding.
3. Return the top-K most similar `DocumentChunk` objects (K configurable,
   default 3).

### FR-4 FastAPI endpoint — ingest
`POST /retrieval/ingest`
- Requires authentication (`READ_DOCUMENTS` permission).
- Triggers `IngestPipeline.run()` on the configured sample-documents directory.
- Embeds all chunks and upserts them into the vector store.
- Returns a count of chunks ingested.

### FR-5 FastAPI endpoint — search
`POST /retrieval/search`
- Requires authentication (`READ_DOCUMENTS` permission).
- Accepts `{ "query": "<question>", "top_k": <int> }`.
- Returns the top-K matching chunks with their similarity scores and metadata.

---

## Non-Functional Requirements

### NFR-1 Swappable embedding provider
The Bedrock Titan implementation must sit behind an `EmbeddingProvider`
interface (Python `Protocol`). Swapping to a different model (OpenAI, Cohere,
local sentence-transformers) shall require only a new adapter class — no
changes to domain or service code.

### NFR-2 Swappable vector store
The JSON-file implementation must sit behind a `VectorStore` interface
(`Protocol`). Migrating to pgvector or Bedrock Knowledge Bases shall require
only a new adapter — no changes to service or domain code.

### NFR-3 Domain layer stays clean
`domain/` must contain zero imports from `boto3`, `json`, `pathlib`, or any
other I/O library. All I/O lives in `infrastructure/`.

### NFR-4 Testability without AWS
Tests must be runnable without AWS credentials. The `EmbeddingProvider`
interface makes it trivial to inject a fake that returns a deterministic
fixed-length vector. Real Bedrock calls are exercised only in optional
integration tests guarded by a `BEDROCK_INTEGRATION=1` environment variable.

### NFR-5 Embedding dimension consistency
All embeddings stored and queried must use the same vector dimension. Titan
Embeddings G1 produces **1536-dimensional** vectors. The system shall raise a
clear error if a stored vector's dimension does not match the query vector.

### NFR-6 Cosine similarity in NumPy
No vector-DB library required for similarity in this phase. `numpy` (already
in `pyproject.toml`) is used for the dot-product and norm computation.

---

## Out of Scope (Phase 3)

- pgvector, Pinecone, or any external vector database (Phase 9)
- LLM answer generation — Phase 3 returns *chunks*, not a synthesised answer (Phase 5)
- Re-ranking, HyDE, or other advanced retrieval techniques (Phase 5+)
- Async Bedrock calls (keep synchronous for now; async wrapper comes in Phase 5)
- PDF/DOCX ingestion (Phase 2 already decided Markdown-only for now)
