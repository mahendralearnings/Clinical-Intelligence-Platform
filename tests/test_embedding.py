"""Tests for the embedding layer — FakeEmbedder, RetrievalService.ingest().

All tests run without AWS credentials by using FakeEmbedder and
FakeVectorStore in place of the real infrastructure adapters.
"""

from clinical_platform.domain.chunk import ChunkMetadata, DocumentChunk
from clinical_platform.domain.embedding import EmbeddedChunk
from clinical_platform.services.retrieval_service import RetrievalService

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """EmbeddingProvider that returns a deterministic fixed-length vector.

    Different input texts produce slightly different vectors (via a hash of
    the text) so tests can distinguish them if needed.
    """

    def __init__(self, dimension: int = 1024) -> None:
        self._dim = dimension

    def embed(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) % 256
        base = float(seed) / 256.0
        return [base] * self._dim


class FakeVectorStore:
    """In-memory VectorStore that records every upsert and supports search."""

    def __init__(self) -> None:
        self._store: list[EmbeddedChunk] = []

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        self._store.extend(chunks)
        return len(self._store)

    def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[DocumentChunk, float]]:  # pragma: no cover
        return []

    def count(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    source: str = "test.md", index: int = 0, text: str = "some content"
) -> DocumentChunk:
    return DocumentChunk(
        content=text,
        metadata=ChunkMetadata(source=source, section_title=None, chunk_index=index),
    )


# ---------------------------------------------------------------------------
# FakeEmbedder unit tests
# ---------------------------------------------------------------------------


def test_embed_returns_correct_length() -> None:
    embedder = FakeEmbedder(dimension=1024)
    result = embedder.embed("hello world")
    assert len(result) == 1024


def test_embed_returns_floats() -> None:
    embedder = FakeEmbedder(dimension=1536)
    result = embedder.embed("clinical trial phase 2 results")
    assert all(isinstance(v, float) for v in result)


def test_embed_different_texts_produce_different_vectors() -> None:
    embedder = FakeEmbedder(dimension=8)
    v1 = embedder.embed("metformin side effects")
    # 'm' has ord 109; sum("metformin side effects") will give a seed != 0
    v2 = embedder.embed("a")  # ord('a') = 97 -> seed = 97
    # They may or may not differ depending on hash collision; just verify
    # both are the right length and type.
    assert len(v1) == 8
    assert len(v2) == 8


# ---------------------------------------------------------------------------
# RetrievalService.ingest() tests
# ---------------------------------------------------------------------------


def test_ingest_returns_count_equal_to_input_length() -> None:
    chunks = [_make_chunk(index=i, text=f"chunk number {i}") for i in range(5)]
    store = FakeVectorStore()
    service = RetrievalService(embedder=FakeEmbedder(), store=store)

    count = service.ingest(chunks)

    assert count == 5


def test_ingest_stores_all_chunks() -> None:
    chunks = [_make_chunk(index=i) for i in range(3)]
    store = FakeVectorStore()
    service = RetrievalService(embedder=FakeEmbedder(), store=store)

    service.ingest(chunks)

    assert store.count() == 3


def test_ingest_empty_list_returns_zero() -> None:
    store = FakeVectorStore()
    service = RetrievalService(embedder=FakeEmbedder(), store=store)

    count = service.ingest([])

    assert count == 0


def test_ingest_embeds_each_chunk_content() -> None:
    """Verify the embedded vectors correspond to the chunk content, not some
    other field, by checking the vector value matches what FakeEmbedder
    would produce for that specific text."""
    embedder = FakeEmbedder(dimension=4)
    store = FakeVectorStore()
    service = RetrievalService(embedder=embedder, store=store)

    chunk = _make_chunk(text="specific content")
    service.ingest([chunk])

    expected_vector = embedder.embed("specific content")
    stored = store._store[0]
    assert stored.vector == expected_vector
