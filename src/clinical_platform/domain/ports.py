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
        self, query_vector: list[float], top_k: int, min_score: float = 0.0
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the *top_k* most similar chunks ranked by cosine similarity.

        Args:
            query_vector: Embedding of the search question.
            top_k:        Maximum number of results to return.
            min_score:    Minimum cosine similarity threshold (default 0.0 —
                          no filter). RagQueryService passes 0.5.

        Returns:
            List of (DocumentChunk, score) tuples, highest score first.

        Raises:
            DimensionMismatchError: if query vector length != stored vector length.
        """
        ...  # pragma: no cover

    def count(self) -> int:
        """Return the total number of stored chunks."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# LLMProvider protocol  (Phase 4)
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised when an LLM provider fails to generate a response.

    Infrastructure adapters catch boto3/HTTP exceptions and re-raise as
    LLMError so service code never has to import infrastructure packages.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Generate a text response from a plain-string prompt.

    Implementations: BedrockLLMProvider (infra), FakeLLMProvider (tests).

    Piece 3 will extend this with system_prompt, max_tokens, and streaming.
    """

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Return a text response for *prompt*.

        Args:
            prompt:        User message to send to the model.
            system_prompt: Optional system-level instruction (e.g. "You are a
                           clinical assistant."). Passed as the ``system``
                           field in Bedrock's converse() API.
            max_tokens:    Maximum tokens in the response (default 512).

        Raises:
            LLMError: if the underlying model call fails for any reason.
        """
        ...  # pragma: no cover
