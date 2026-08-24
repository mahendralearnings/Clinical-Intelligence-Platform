"""Response shape for the observability summary endpoint."""

from pydantic import BaseModel


class ObservabilitySummary(BaseModel):
    total_queries: int
    answered_from_documents: int
    said_i_dont_know: int
    average_top_score: float
    most_common_source_document: str | None