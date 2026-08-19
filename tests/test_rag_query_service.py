"""Tests for RagQueryService — Phase 5.

All tests use fakes — no AWS calls, no real embeddings, no real files
beyond pytest's tmp_path fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

from clinical_platform.domain.chunk import ChunkMetadata, DocumentChunk
from clinical_platform.domain.embedding import EmbeddedChunk
from clinical_platform.domain.rag import RagResult
from clinical_platform.services.rag_query_service import RagQueryService
from clinical_platform.services.retrieval_service import RetrievalService

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Returns a deterministic fixed-length vector — no AWS calls."""

    def __init__(self, dimension: int = 4) -> None:
        self._dim = dimension

    def embed(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) % 256
        return [float(seed) / 256.0] * self._dim


class FakeVectorStore:
    """Returns pre-configured (chunk, score) pairs, respects min_score."""

    def __init__(self, results: list[tuple[DocumentChunk, float]]) -> None:
        self._results = results

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[tuple[DocumentChunk, float]]:
        filtered = [(c, s) for c, s in self._results if s >= min_score]
        return filtered[:top_k]

    def count(self) -> int:
        return len(self._results)


class FakeLLMProvider:
    """Returns a fixed string; tracks whether generate() was ever called."""

    def __init__(self, response: str = "The answer is metformin.") -> None:
        self._response = response
        self.called = False

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        self.called = True
        return self._response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(source: str, section: str | None, text: str) -> DocumentChunk:
    return DocumentChunk(
        content=text,
        metadata=ChunkMetadata(source=source, section_title=section, chunk_index=0),
    )


def _make_service(
    store_results: list[tuple[DocumentChunk, float]],
    llm: FakeLLMProvider,
    log_path: Path,
) -> RagQueryService:
    retrieval = RetrievalService(
        embedder=FakeEmbedder(),
        store=FakeVectorStore(store_results),
    )
    return RagQueryService(retrieval=retrieval, llm=llm, log_path=log_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_query_returns_answer_and_correct_sources(tmp_path: Path) -> None:
    """Good matching chunks -> LLM is called -> answer + citations returned."""
    chunk_a = _chunk("drug_manual_metformin.md", "Side Effects", "lactic acidosis risk")
    chunk_b = _chunk("sop_adverse_event_reporting.md", "4.1", "report within 24 hours")
    llm = FakeLLMProvider(response="Metformin can cause lactic acidosis.")

    service = _make_service(
        store_results=[(chunk_a, 0.92), (chunk_b, 0.75)],
        llm=llm,
        log_path=tmp_path / "log.jsonl",
    )

    result = service.query("What are the side effects of metformin?", top_k=3)

    assert isinstance(result, RagResult)
    assert result.answer == "Metformin can cause lactic acidosis."
    assert len(result.sources) == 2
    assert result.sources[0].source == "drug_manual_metformin.md"
    assert result.sources[0].section_title == "Side Effects"
    assert result.sources[0].score == 0.92
    assert llm.called is True


def test_no_chunks_above_min_score_returns_i_dont_know_without_llm_call(
    tmp_path: Path,
) -> None:
    """Chunks below min_score=0.5 -> 'I don't know' returned, LLM never called.

    This proves we skip the wasted AI call when no relevant context exists.
    """
    chunk = _chunk("some_doc.md", None, "irrelevant content")
    llm = FakeLLMProvider()

    service = _make_service(
        store_results=[(chunk, 0.3)],  # below min_score threshold of 0.5
        llm=llm,
        log_path=tmp_path / "log.jsonl",
    )

    result = service.query("What is lactic acidosis?")

    assert "I don't know" in result.answer
    assert result.sources == []
    assert llm.called is False  # critical: no wasted LLM call


def test_log_file_gets_one_line_per_query(tmp_path: Path) -> None:
    """Each query() call appends exactly one valid JSON line to the log."""
    chunk = _chunk("drug_manual_metformin.md", "Dosage", "take 500mg twice daily")
    llm = FakeLLMProvider(response="Take 500mg twice daily.")

    service = _make_service(
        store_results=[(chunk, 0.85)],
        llm=llm,
        log_path=tmp_path / "log.jsonl",
    )

    service.query("What is the metformin dosage?")
    service.query("Any other dosage info?")

    log_path = tmp_path / "log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    for line in lines:
        record = json.loads(line)
        assert "question" in record
        assert "answer" in record
        assert "timestamp" in record
        assert "retrieved_chunks" in record
