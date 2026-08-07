"""Domain value object for an embedded document chunk.

Pure Python — no boto3, no file I/O, no framework imports.
"""

from dataclasses import dataclass

from clinical_platform.domain.chunk import DocumentChunk


@dataclass(frozen=True)
class EmbeddedChunk:
    """A DocumentChunk paired with its embedding vector.

    Keeping the original ``chunk`` intact (rather than flattening its fields)
    means callers always have the full ``DocumentChunk`` available without
    needing to re-parse stored JSON.
    """

    chunk: DocumentChunk
    vector: list[float]
