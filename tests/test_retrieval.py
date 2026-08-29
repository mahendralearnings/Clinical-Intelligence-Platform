"""Tests for the retrieval (search) layer.

All tests run without AWS credentials by using crafted embedding vectors
and JsonVectorStore backed by a temp file.

The key semantic test (lactic acidosis -> metformin) uses carefully crafted
vectors to prove the cosine-similarity ranking is correct without real
Bedrock calls.  An optional integration variant (guarded by
BEDROCK_INTEGRATION=1) exercises the real Titan model end-to-end.
"""

import os
from pathlib import Path

import pytest

from clinical_platform.domain.chunk import ChunkMetadata, DocumentChunk
from clinical_platform.domain.embedding import EmbeddedChunk
from clinical_platform.domain.ports import DimensionMismatchError
from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
from clinical_platform.services.retrieval_service import RetrievalService

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Returns a pre-set vector for each input text, falling back to zeros."""

    def __init__(self, mapping: dict[str, list[float]], dimension: int = 8) -> None:
        self._mapping = mapping
        self._dim = dimension

    def embed(self, text: str) -> list[float]:
        return self._mapping.get(text, [0.0] * self._dim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    source: str = "test.md",
    index: int = 0,
    text: str = "content",
    section: str | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        content=text,
        metadata=ChunkMetadata(source=source, section_title=section, chunk_index=index),
    )


def _make_embedded(
    chunk: DocumentChunk,
    vector: list[float],
) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=chunk, vector=vector)


# ---------------------------------------------------------------------------
# JsonVectorStore.search() — cosine ranking
# ---------------------------------------------------------------------------


def test_search_returns_most_relevant_chunk(tmp_path: Path) -> None:
    """A chunk whose vector is aligned with the query should rank above one
    that is orthogonal to it."""
    store = JsonVectorStore(store_path=tmp_path / "vs.json")

    # query vector points along dimension 0
    query_vector = [1.0, 0.0, 0.0, 0.0]

    # 'close' chunk: vector also points along dimension 0 -> cosine sim = 1.0
    close_chunk = _make_chunk(source="drug_manual_metformin.md", index=0, text="metformin content")
    close_embedded = _make_embedded(close_chunk, [1.0, 0.0, 0.0, 0.0])

    # 'far' chunk: vector is orthogonal -> cosine sim = 0.0
    far_chunk = _make_chunk(source="sop_adverse_event_reporting.md", index=0, text="SOP content")
    far_embedded = _make_embedded(far_chunk, [0.0, 1.0, 0.0, 0.0])

    store.upsert([close_embedded, far_embedded])

    results = store.search(query_vector, top_k=2)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.metadata.source == "drug_manual_metformin.md"
    assert top_score > 0.99


def test_lactic_acidosis_returns_metformin_chunk(tmp_path: Path) -> None:
    """Simulate the semantic test: 'lactic acidosis' question should surface
    the metformin chunk, not the SOP chunk.

    Uses crafted vectors that mimic what a real embedding model would produce:
    both the question and the metformin chunk share high weight on a
    'pharmacology' dimension; the SOP chunk does not.
    """
    store = JsonVectorStore(store_path=tmp_path / "vs.json")

    # Dimension layout (conceptual): [pharma, procedure, trial, other]
    # Metformin chunk: high pharma weight
    metformin_chunk = _make_chunk(
        source="drug_manual_metformin.md",
        index=2,
        text="Lactic acidosis is a rare but serious side effect of metformin.",
        section="Side Effects",
    )
    # SOP chunk: high procedure weight, low pharma weight
    sop_chunk = _make_chunk(
        source="sop_adverse_event_reporting.md",
        index=0,
        text="Step 1: Document the adverse event in the eCRF within 24 hours.",
        section="4.1 Identification and Initial Documentation",
    )

    store.upsert([
        _make_embedded(metformin_chunk, [0.9, 0.1, 0.1, 0.1]),
        _make_embedded(sop_chunk,       [0.1, 0.9, 0.1, 0.1]),
    ])

    # Query about lactic acidosis: high pharma weight
    lactic_acidosis_query_vector = [0.85, 0.05, 0.05, 0.05]
    results = store.search(lactic_acidosis_query_vector, top_k=1)

    assert len(results) == 1
    top_chunk, _ = results[0]
    assert top_chunk.metadata.source == "drug_manual_metformin.md", (
        "Searching for 'lactic acidosis' should surface the metformin chunk, not the SOP."
    )


def test_search_top_k_respected(tmp_path: Path) -> None:
    """Requesting top_k=2 from a store with 5 chunks returns exactly 2."""
    store = JsonVectorStore(store_path=tmp_path / "vs.json")

    chunks = [
        _make_embedded(
            _make_chunk(index=i, text=f"chunk {i}"),
            [float(i), 0.0, 0.0, 0.0],
        )
        for i in range(5)
    ]
    store.upsert(chunks)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2


def test_search_empty_store_returns_empty_list(tmp_path: Path) -> None:
    store = JsonVectorStore(store_path=tmp_path / "vs.json")
    results = store.search([1.0, 0.0, 0.0], top_k=3)
    assert results == []


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    """Querying with a different vector dimension than what is stored must
    raise DimensionMismatchError — not a silent wrong answer."""
    store = JsonVectorStore(store_path=tmp_path / "vs.json")

    store.upsert([_make_embedded(_make_chunk(), [0.1] * 1024)])

    with pytest.raises(DimensionMismatchError, match="dimension"):
        store.search([0.1] * 512, top_k=1)


# ---------------------------------------------------------------------------
# RetrievalService.search() — end-to-end with fake embedder + real JSON store
# ---------------------------------------------------------------------------


def test_service_search_returns_correct_source(tmp_path: Path) -> None:
    """Wire RetrievalService with a FakeEmbedder that returns known vectors,
    then confirm search() returns the right chunk."""
    metformin_text = "metformin lactic acidosis risk"
    sop_text = "adverse event reporting steps"
    query_text = "what is the risk of lactic acidosis?"

    # metformin vector: [1, 0, 0, 0]; SOP vector: [0, 1, 0, 0]; query: [1, 0, 0, 0]
    embedder = FakeEmbedder(
        mapping={
            metformin_text: [1.0, 0.0, 0.0, 0.0],
            sop_text:       [0.0, 1.0, 0.0, 0.0],
            query_text:     [1.0, 0.0, 0.0, 0.0],
        },
        dimension=4,
    )
    store = JsonVectorStore(store_path=tmp_path / "vs.json")
    service = RetrievalService(embedder=embedder, store=store)

    metformin_chunk = _make_chunk(source="drug_manual_metformin.md", text=metformin_text)
    sop_chunk = _make_chunk(source="sop_adverse_event_reporting.md", text=sop_text)
    service.ingest([metformin_chunk, sop_chunk])

    results = service.search(query_text, top_k=1)

    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.metadata.source == "drug_manual_metformin.md"
    assert score > 0.99


def test_service_search_top_k_flows_through(tmp_path: Path) -> None:
    """top_k parameter is forwarded from service to store correctly."""
    chunks = [_make_chunk(index=i, text=f"doc {i}") for i in range(10)]

    embedder = FakeEmbedder(
        mapping={f"doc {i}": [float(i % 2), float((i + 1) % 2), 0.0, 0.0] for i in range(10)},
        dimension=4,
    )
    store = JsonVectorStore(store_path=tmp_path / "vs.json")
    service = RetrievalService(embedder=embedder, store=store)
    service.ingest(chunks)

    results = service.search("doc 0", top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# JsonVectorStore upsert deduplication
# ---------------------------------------------------------------------------


def test_upsert_deduplicates_by_source_and_index(tmp_path: Path) -> None:
    """Re-ingesting the same (source, chunk_index) key replaces the entry
    rather than creating a duplicate."""
    store = JsonVectorStore(store_path=tmp_path / "vs.json")

    chunk = _make_chunk(source="doc.md", index=0, text="original text")
    store.upsert([_make_embedded(chunk, [1.0, 0.0])])
    assert store.count() == 1

    updated_chunk = _make_chunk(source="doc.md", index=0, text="updated text")
    store.upsert([_make_embedded(updated_chunk, [0.9, 0.1])])

    # Still only 1 entry — not 2
    assert store.count() == 1

    # The stored content should be the updated version
    results = store.search([1.0, 0.0], top_k=1)
    assert results[0][0].content == "updated text"


# ---------------------------------------------------------------------------
# Optional real-Bedrock integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("BEDROCK_INTEGRATION") != "1",
    reason="Set BEDROCK_INTEGRATION=1 to run real AWS Bedrock calls",
)
def test_bedrock_lactic_acidosis_returns_metformin_chunk(  # pragma: no cover
    tmp_path: Path,
) -> None:
    """End-to-end test using the real Titan embedding model.

    Prerequisites:
        - Valid AWS credentials in environment / ~/.aws/credentials
        - Bedrock Titan Embeddings enabled in us-east-1
        - BEDROCK_INTEGRATION=1 environment variable set
    """
    from pathlib import Path as _Path

    from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
        BedrockEmbeddingProvider,
    )
    from clinical_platform.infrastructure.ingestion.pipeline import IngestPipeline

    docs_dir = _Path(__file__).parent.parent / "data" / "sample_documents"
    embedder = BedrockEmbeddingProvider()
    store = JsonVectorStore(store_path=tmp_path / "vs_integration.json")
    service = RetrievalService(embedder=embedder, store=store)

    pipeline = IngestPipeline(docs_dir=docs_dir)
    chunks = pipeline.run()
    service.ingest(chunks)

    results = service.search("What is the risk of lactic acidosis?", top_k=3)

    assert results, "Expected at least one result"
    sources = [chunk.metadata.source for chunk, _ in results]
    assert "drug_manual_metformin.md" in sources, (
        f"Expected metformin chunk in top-3 results, got sources: {sources}"
    )
