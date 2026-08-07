"""Retrieval service — orchestrates embedding and vector storage.

This is the only place that wires EmbeddingProvider and VectorStore together.
It has zero imports from boto3, json, pathlib, or any infrastructure package.
All I/O is delegated to the injected adapters.
"""

from clinical_platform.domain.chunk import DocumentChunk
from clinical_platform.domain.embedding import EmbeddedChunk
from clinical_platform.domain.ports import EmbeddingProvider, VectorStore


class RetrievalService:
    """High-level facade for ingest and semantic search operations.

    Depends only on the two domain Protocols — swap any implementation
    (Bedrock -> OpenAI, JSON file -> pgvector) without touching this class.

    Args:
        embedder: Anything satisfying the EmbeddingProvider protocol.
        store:    Anything satisfying the VectorStore protocol.
    """

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> int:
        """Embed every chunk and upsert into the vector store.

        Args:
            chunks: Flat list produced by ``IngestPipeline.run()``.

        Returns:
            Number of chunks successfully written to the store.
        """
        embedded: list[EmbeddedChunk] = [
            EmbeddedChunk(chunk=chunk, vector=self._embedder.embed(chunk.content))
            for chunk in chunks
        ]
        return self._store.upsert(embedded)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 3
    ) -> list[tuple[DocumentChunk, float]]:
        """Embed *query* and return the top-K most similar stored chunks.

        Args:
            query:  Natural-language question from the caller.
            top_k:  Maximum number of results to return (default 3).

        Returns:
            List of (DocumentChunk, cosine_similarity_score) tuples,
            ordered highest score first.

        Raises:
            DimensionMismatchError: propagated from the VectorStore if the
                query vector dimension does not match stored vectors.
            EmbeddingError: propagated from the EmbeddingProvider if the
                model call fails.
        """
        query_vector = self._embedder.embed(query)
        return self._store.search(query_vector, top_k)
