"""Domain model for a document chunk produced after splitting.

Pure Python — no FastAPI, no LangChain, no I/O.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMetadata:
    source: str               # filename, e.g. "sop_adverse_event_reporting.md"
    section_title: str | None # nearest ## or ### header text, or None
    chunk_index: int          # zero-based, unique within a single document


@dataclass(frozen=True)
class DocumentChunk:
    content: str
    metadata: ChunkMetadata
