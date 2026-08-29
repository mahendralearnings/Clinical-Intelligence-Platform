"""Domain value objects for the RAG Query Service (Phase 5).

Pure Python — no I/O, no boto3, no framework imports.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceCitation:
    """One retrieved chunk used to ground the answer.

    Returned alongside the answer so callers can show citations.
    """

    source: str                # filename e.g. "drug_manual_metformin.md"
    section_title: str | None  # nearest ## / ### header, or None
    score: float               # cosine similarity score in [-1, 1]


@dataclass(frozen=True)
class RagResult:
    """Full output of a RAG query.

    answer  — Claude's response, grounded strictly in retrieved chunks.
    sources — citations for each chunk passed to the LLM.
              Empty list when no chunks cleared the min_score threshold.
    """

    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
