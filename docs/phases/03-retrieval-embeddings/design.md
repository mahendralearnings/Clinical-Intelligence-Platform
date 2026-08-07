# Phase 3 — Retrieval & Embeddings: Design

## 1. Architecture Overview

Phase 3 follows the same hexagonal pattern established in Phases 0-2.

```
┌─────────────────── domain/ ────────────────────────────────┐
│  EmbeddingProvider (Protocol)                               │
│  VectorStore       (Protocol)                               │
│  EmbeddingError    (exception)                              │
│  DimensionMismatchError (exception)                         │
│  EmbeddedChunk     (dataclass)  ← new domain value object   │
└─────────────────────────────────────────────────────────────┘
            ↑ depends on              ↑ depends on
┌───────── services/ ────────────────────────────────────────┐
│  RetrievalService                                           │
│    .ingest(chunks)  → calls EmbeddingProvider + VectorStore │
│    .search(query, k) → calls EmbeddingProvider + VectorStore│
└─────────────────────────────────────────────────────────────┘
            ↑ wired by               ↑ wired by
┌───────── infrastructure/ ──────────────────────────────────┐
│  BedrockEmbeddingProvider  (implements EmbeddingProvider)   │
│  JsonVectorStore           (implements VectorStore)         │
└─────────────────────────────────────────────────────────────┘
            ↑ exposed via
┌───────── api/routes/ ──────────────────────────────────────┐
│  retrieval.py  (POST /retrieval/ingest, /retrieval/search)  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. New Domain Models

### `domain/embedding.py`

```python
@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: DocumentChunk     # the original Phase 2 chunk (content + metadata)
    vector: list[float]      # embedding produced by EmbeddingProvider
```

`EmbeddedChunk` is the single value object that crosses the boundary between
the embedding step and the storage step. Keeping `chunk` intact (rather than
flattening fields) means the original `DocumentChunk` is always recoverable
without re-parsing stored JSON.

### `domain/ports.py`

Both protocols live in one file — they are small (3 methods combined) and are
always imported together by `RetrievalService`. One file avoids circular-import
risk and keeps the interface surface easy to review.

```python
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

class VectorStore(Protocol):
    def upsert(self, chunks: list[EmbeddedChunk]) -> int: ...
    def search(self, query_vector: list[float], top_k: int) -> list[tuple[DocumentChunk, float]]: ...
    def count(self) -> int: ...
```

Domain exceptions also live here so infrastructure can raise them without
leaking `botocore` or `IOError` into service code:

```python
class EmbeddingError(Exception): ...
class DimensionMismatchError(Exception): ...
```

**Why `Protocol` instead of `ABC`?**
Structural subtyping — test doubles satisfy the interface without inheriting
from a base class. This means future third-party adapters work without
modification to domain code.

---

## 3. Service Layer

### `services/retrieval_service.py`

```python
class RetrievalService:
    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None: ...

    def ingest(self, chunks: list[DocumentChunk]) -> int:
        """Embed each chunk and upsert into the store. Returns count."""

    def search(self, query: str, top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
        """Embed query, call store.search(), return ranked chunks + scores."""
```

`RetrievalService` imports **nothing** from `boto3`, `json`, or `pathlib`.
It only knows about the two Protocols. Fully unit-testable with fakes injected
at construction time.

---

## 4. Infrastructure Adapters

### `infrastructure/embedding_providers/bedrock_embedding_provider.py`

- Wraps `boto3.client("bedrock-runtime")`.
- Calls `invoke_model` with model ID `amazon.titan-embed-text-v1`.
- Parses the response JSON to extract `embedding` (list of 1536 floats).
- Raises `EmbeddingError` (from `domain/ports.py`) on AWS errors — never lets
  `botocore.exceptions` propagate into service code.

**Titan Embeddings G1 — key facts:**
- Model ID: `amazon.titan-embed-text-v1`
- Output dimension: 1536
- Max input tokens: 8192
- Cost: ~$0.0001 per 1K tokens

### `infrastructure/vector_stores/json_vector_store.py`

- Reads/writes a single JSON file (default `data/vector_store.json`).
- File schema — a JSON array, one object per `EmbeddedChunk`:
  ```json
  [
    {
      "content": "...",
      "source": "drug_manual_metformin.md",
      "section_title": "Side Effects",
      "chunk_index": 3,
      "vector": [0.012, -0.034, ...]
    }
  ]
  ```
- `upsert()`: loads existing file (or starts empty), deduplicates by
  `(source, chunk_index)`, writes back atomically (write `.tmp` then rename).
- `search()`: loads file, computes cosine similarity for each entry using
  `numpy`, sorts descending, returns top-K `(DocumentChunk, score)` tuples.
- `count()`: returns `len` of the loaded array.
- Raises `DimensionMismatchError` if query vector length ≠ stored vector length.

**Why JSON and not SQLite?**
SQLite requires schema migrations and is overkill before a real vector DB
arrives in Phase 9. JSON is readable, grep-able, and inspectable during
interviews. The `VectorStore` interface means swapping it costs zero
service-layer changes.

---

## 5. API Layer

### `api/schemas/retrieval.py` — Pydantic request/response models

| Schema | Fields |
|--------|--------|
| `IngestRequest` | `docs_dir: str \| None` (optional override) |
| `IngestResponse` | `chunks_ingested: int` |
| `SearchRequest` | `query: str`, `top_k: int = 3` |
| `SearchResultItem` | `content`, `source`, `section_title`, `chunk_index`, `score` |
| `SearchResponse` | `results: list[SearchResultItem]` |

### `api/routes/retrieval.py`

Both endpoints gated on `Permission.READ_DOCUMENTS`:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/retrieval/ingest` | Run pipeline, embed, upsert. Returns `{ chunks_ingested }` |
| `POST` | `/retrieval/search` | Embed query, cosine search. Returns `{ results: [...] }` |

Dependency injection via `Depends(get_retrieval_service)` — same pattern as
`AuthService`.

---

## 6. Similarity Computation

Cosine similarity: `dot(a, b) / (norm(a) * norm(b))`

Implemented with `numpy` inside `JsonVectorStore.search()` — it is an
implementation detail of this specific store, not a domain concern.

---

## 7. Configuration — new fields in `core/config.py`

| Field | Type | Default |
|-------|------|---------|
| `docs_dir` | `str` | `data/sample_documents` |
| `vector_store_path` | `str` | `data/vector_store.json` |
| `bedrock_region` | `str` | `us-east-1` |
| `embedding_model_id` | `str` | `amazon.titan-embed-text-v1` |
| `embedding_dimension` | `int` | `1536` |

---

## 8. New Dependencies

| Package | Why | Added to |
|---------|-----|---------|
| `boto3>=1.35.0` | Bedrock Titan API calls | `pyproject.toml` `dependencies` |

`numpy` is already present in `pyproject.toml`.

---

## 9. Test Plan

### `tests/test_embedding.py`

| Test | Approach | Assert |
|------|----------|--------|
| `test_embed_returns_correct_length` | `FakeEmbedder` (returns fixed 1536-d vector) | `len(result) == 1536` |
| `test_embed_returns_floats` | Same fake | all elements are `float` |
| `test_ingest_count` | `RetrievalService` + `FakeEmbedder` + `FakeVectorStore` | count equals input length |

### `tests/test_retrieval.py`

| Test | Approach | Assert |
|------|----------|--------|
| `test_search_returns_most_relevant_chunk` | Two `EmbeddedChunk`s with crafted vectors; one close to query, one far | top result is the close chunk |
| `test_lactic_acidosis_returns_metformin_chunk` | Real Titan (skipped unless `BEDROCK_INTEGRATION=1`) or crafted vectors | `result[0].metadata.source == "drug_manual_metformin.md"` |
| `test_search_top_k_respected` | 5 chunks stored, `top_k=2` | exactly 2 results |
| `test_dimension_mismatch_raises` | Store 1536-d vector, query with 768-d | raises `DimensionMismatchError` |

### `FakeEmbedder` strategy

```python
class FakeEmbedder:
    def __init__(self, dimension: int = 1536):
        self._dim = dimension

    def embed(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) % 256
        return [float(seed) / 256.0] * self._dim
```

Satisfies `EmbeddingProvider` structurally — no inheritance needed.

---

## 10. File Changeset Summary

### New files
```
docs/phases/03-retrieval-embeddings/requirements.md
docs/phases/03-retrieval-embeddings/design.md
src/clinical_platform/domain/embedding.py
src/clinical_platform/domain/ports.py
src/clinical_platform/services/retrieval_service.py
src/clinical_platform/infrastructure/embedding_providers/__init__.py
src/clinical_platform/infrastructure/embedding_providers/bedrock_embedding_provider.py
src/clinical_platform/infrastructure/vector_stores/json_vector_store.py
src/clinical_platform/api/schemas/retrieval.py
src/clinical_platform/api/routes/retrieval.py
tests/test_embedding.py
tests/test_retrieval.py
```

### Modified files
```
src/clinical_platform/main.py          ← include retrieval router
src/clinical_platform/core/config.py   ← add 5 new settings fields
pyproject.toml                         ← add boto3 dependency
```
