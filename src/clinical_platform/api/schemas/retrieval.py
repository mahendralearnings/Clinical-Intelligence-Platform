"""Pydantic schemas for retrieval request and response bodies."""

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Optional override for the documents directory.

    If omitted the service uses ``Settings.docs_dir``.
    """

    docs_dir: str | None = Field(
        default=None,
        description="Absolute or relative path to the directory of .md/.txt files to ingest.",
    )


class IngestResponse(BaseModel):
    chunks_ingested: int = Field(description="Total chunks now present in the vector store.")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language question to search for.")
    top_k: int = Field(default=3, ge=1, le=20, description="Maximum number of results to return.")


class SearchResultItem(BaseModel):
    content: str
    source: str
    section_title: str | None
    chunk_index: int
    score: float = Field(description="Cosine similarity score in [-1, 1].")


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
