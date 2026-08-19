"""JSON-file implementation of the VectorStore protocol.

Stores all EmbeddedChunks as a JSON array on disk.  Intended for local
development and interview demos — swap for pgvector in Phase 9 by writing
a new class that satisfies the same VectorStore protocol.

Design decisions:
- Atomic writes: write to a .tmp file then os.replace() to avoid corrupt
  state if the process is killed mid-write.
- Deduplication key: (source, chunk_index) — re-ingesting the same document
  replaces existing entries rather than creating duplicates.
- Cosine similarity via numpy (already a project dependency).
- DimensionMismatchError is raised before any similarity is computed if the
  query vector length does not match the stored vectors' length.
"""

import json
import os
from pathlib import Path

import numpy as np

from clinical_platform.domain.chunk import ChunkMetadata, DocumentChunk
from clinical_platform.domain.embedding import EmbeddedChunk
from clinical_platform.domain.ports import DimensionMismatchError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_to_record(ec: EmbeddedChunk) -> dict[str, object]:
    """Flatten an EmbeddedChunk to a JSON-serialisable dict."""
    return {
        "content": ec.chunk.content,
        "source": ec.chunk.metadata.source,
        "section_title": ec.chunk.metadata.section_title,
        "chunk_index": ec.chunk.metadata.chunk_index,
        "vector": ec.vector,
    }


def _record_to_chunk(record: dict[str, object]) -> DocumentChunk:
    """Reconstruct a DocumentChunk from a stored JSON record."""
    return DocumentChunk(
        content=str(record["content"]),
        metadata=ChunkMetadata(
            source=str(record["source"]),
            section_title=(
                str(record["section_title"])
                if record["section_title"] is not None
                else None
            ),
            chunk_index=int(str(record["chunk_index"])),
        ),
    )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [-1, 1] between vectors *a* and *b*."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# JsonVectorStore
# ---------------------------------------------------------------------------

class JsonVectorStore:
    """VectorStore backed by a single JSON file.

    Args:
        store_path: Path to the JSON file.  Created on first upsert if it
            does not already exist.
    """

    def __init__(self, store_path: Path) -> None:
        self._path = store_path

    # ------------------------------------------------------------------
    # Internal load / save
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, object]]:
        """Return the stored records, or an empty list if the file is absent."""
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return []
        return data  # json.load returns Any; mypy infers list[Any] which satisfies our return type

    def _save(self, records: list[dict[str, object]]) -> None:
        """Write *records* atomically: write to .tmp, then os.replace()."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        """Insert or replace chunks, keyed by (source, chunk_index).

        Returns:
            Total number of chunks in the store after the upsert.
        """
        existing = self._load()

        # Build an ordered dict so we can replace by key while preserving
        # order for older entries and appending new ones.
        keyed: dict[tuple[str, int], dict[str, object]] = {
            (str(r["source"]), int(str(r["chunk_index"]))): r
            for r in existing
        }

        for ec in chunks:
            key = (ec.chunk.metadata.source, ec.chunk.metadata.chunk_index)
            keyed[key] = _chunk_to_record(ec)

        records = list(keyed.values())
        self._save(records)
        return len(records)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Return the top-K chunks most similar to *query_vector*.

        Args:
            query_vector: Embedding of the search question.
            top_k:        Maximum number of results to return.
            min_score:    Minimum cosine similarity threshold. Chunks scoring
                          below this value are excluded. Default 0.0 (no filter).

        Raises:
            DimensionMismatchError: if query dimension != stored dimension.
        """
        records = self._load()
        if not records:
            return []

        # Dimension guard — check against the first stored vector
        first_vector: list[float] = records[0]["vector"]  # type: ignore[assignment]
        if len(query_vector) != len(first_vector):
            raise DimensionMismatchError(
                f"Query vector has dimension {len(query_vector)} but store "
                f"contains dimension {len(first_vector)}."
            )

        scored: list[tuple[DocumentChunk, float]] = [
            (
                _record_to_chunk(r),
                _cosine_similarity(query_vector, r["vector"]),  # type: ignore[arg-type]
            )
            for r in records
        ]

        scored.sort(key=lambda x: x[1], reverse=True)
        # Apply min_score filter after sorting so top_k counts only passing chunks
        filtered = [(chunk, score) for chunk, score in scored if score >= min_score]
        return filtered[:top_k]

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._load())
