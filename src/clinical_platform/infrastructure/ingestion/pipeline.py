"""Ingestion pipeline: wires DocumentLoader and HybridChunker together."""

from pathlib import Path

from clinical_platform.domain.chunk import DocumentChunk
from clinical_platform.infrastructure.ingestion.chunker import HybridChunker
from clinical_platform.infrastructure.ingestion.loader import DocumentLoader


class IngestPipeline:
    """Single entry point for loading and chunking all documents in a folder."""

    def __init__(
        self,
        docs_dir: Path,
        max_chunk_size: int = 500,
        chunk_overlap: int = 75,
    ) -> None:
        self._loader = DocumentLoader(docs_dir)
        self._chunker = HybridChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def run(self) -> list[DocumentChunk]:
        """Load all documents and return a flat list of chunks."""
        all_chunks: list[DocumentChunk] = []
        for doc in self._loader.load():
            all_chunks.extend(self._chunker.chunk(doc))
        return all_chunks
