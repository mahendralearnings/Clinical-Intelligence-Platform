"""Domain-level interfaces (Protocols) and exceptions for Phase 3.

Both protocols live here because they are small and always imported together
by RetrievalService.  One file also avoids any circular-import risk.

No I/O, no boto3, no file-system code — pure domain contracts.
"""

from typing import Protocol, runtime_checkable

from clinical_platform.domain.chunk import DocumentChunk
from clinical_platform.domain.embedding import EmbeddedChunk

# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Raised when an embedding provider fails to produce a vector.

    Infrastructure adapters (e.g. BedrockEmbeddingProvider) catch SDK-specific
    errors (botocore exceptions, HTTP errors) and re-raise as EmbeddingError so
    service code never has to import infrastructure packages.
    """


class DimensionMismatchError(Exception):
    """Raised when a query vector's dimension differs from stored vectors.

    Example message:
        "Query vector has dimension 768 but store contains dimension 1536."
    """


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Convert a text string into a fixed-length float vector.

    Implementations: BedrockEmbeddingProvider (infra), FakeEmbedder (tests).
    """

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for *text*.

        Raises:
            EmbeddingError: if the underlying model call fails.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# VectorStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Persist and retrieve EmbeddedChunks.

    Implementations: JsonVectorStore (infra), FakeVectorStore (tests).
    """

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        """Store *chunks*, replacing any existing entry with the same
        (source, chunk_index) key.

        Returns:
            Number of chunks written (after deduplication).
        """
        ...  # pragma: no cover

    def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the *top_k* most similar chunks ranked by cosine similarity.

        Args:
            query_vector: Embedding of the search question.
            top_k: Maximum number of results to return.

        Returns:
            List of (DocumentChunk, score) tuples, highest score first.

        Raises:
            DimensionMismatchError: if query vector length != stored vector length.
        """
        ...  # pragma: no cover

    def count(self) -> int:
        """Return the total number of stored chunks."""
        ...  # pragma: no cover
