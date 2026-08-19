"""Pydantic schemas for the RAG query endpoint — Phase 5."""

from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, description="Natural-language question.")
    top_k: int = Field(default=3, ge=1, le=20, description="Chunks to retrieve.")


class SourceCitationResponse(BaseModel):
    source: str
    section_title: str | None
    score: float = Field(description="Cosine similarity score in [-1, 1].")


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitationResponse]
