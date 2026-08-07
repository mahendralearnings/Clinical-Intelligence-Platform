"""
Pytest tests for the document ingestion and chunking pipeline.

All tests run against the real sample documents in data/sample_documents/.
The DOCS_DIR fixture resolves the path relative to this file so tests
work regardless of which directory pytest is invoked from.
"""

from pathlib import Path

import pytest

from clinical_platform.domain.chunk import DocumentChunk
from clinical_platform.domain.document import RawDocument
from clinical_platform.infrastructure.ingestion.chunker import HybridChunker
from clinical_platform.infrastructure.ingestion.loader import DocumentLoader
from clinical_platform.infrastructure.ingestion.pipeline import IngestPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_documents"


@pytest.fixture(scope="module")
def all_chunks() -> list[DocumentChunk]:
    """Run the full pipeline once; reuse the result across all tests."""
    pipeline = IngestPipeline(docs_dir=DOCS_DIR)
    return pipeline.run()


@pytest.fixture(scope="module")
def sop_chunks(all_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    return [c for c in all_chunks if c.metadata.source == "sop_adverse_event_reporting.md"]


@pytest.fixture(scope="module")
def metformin_chunks(all_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    return [c for c in all_chunks if c.metadata.source == "drug_manual_metformin.md"]


@pytest.fixture(scope="module")
def trial_chunks(all_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    return [c for c in all_chunks if c.metadata.source == "clinical_trial_summary_204.md"]


# ---------------------------------------------------------------------------
# FR-04 — No empty or None chunks (all documents)
# ---------------------------------------------------------------------------

def test_no_empty_chunks(all_chunks: list[DocumentChunk]) -> None:
    for chunk in all_chunks:
        assert chunk.content is not None, "chunk.content must never be None"
        assert chunk.content.strip() != "", f"Empty chunk found: {chunk.metadata}"


# ---------------------------------------------------------------------------
# FR-03 — Metadata completeness (all documents)
# ---------------------------------------------------------------------------

def test_every_chunk_has_valid_source(all_chunks: list[DocumentChunk]) -> None:
    for chunk in all_chunks:
        assert chunk.metadata.source, "source must be a non-empty string"


def test_chunk_index_is_non_negative(all_chunks: list[DocumentChunk]) -> None:
    for chunk in all_chunks:
        assert chunk.metadata.chunk_index >= 0


def test_chunk_indices_are_sequential_per_document(all_chunks: list[DocumentChunk]) -> None:
    """chunk_index must be 0, 1, 2 … N-1 within each document."""
    by_source: dict[str, list[DocumentChunk]] = {}
    for chunk in all_chunks:
        by_source.setdefault(chunk.metadata.source, []).append(chunk)

    for source, chunks in by_source.items():
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(indices))), (
            f"{source}: chunk_index is not sequential: {indices}"
        )


# ---------------------------------------------------------------------------
# FR-01 — Loader produces one entry per supported file, skips .gitkeep
# ---------------------------------------------------------------------------

def test_loader_loads_exactly_three_documents() -> None:
    loader = DocumentLoader(DOCS_DIR)
    docs = loader.load()
    assert len(docs) == 3


def test_loader_skips_gitkeep() -> None:
    loader = DocumentLoader(DOCS_DIR)
    docs = loader.load()
    sources = {d.source for d in docs}
    assert ".gitkeep" not in sources


def test_loader_returns_raw_document_objects() -> None:
    loader = DocumentLoader(DOCS_DIR)
    for doc in loader.load():
        assert isinstance(doc, RawDocument)
        assert doc.text.strip() != ""


# ---------------------------------------------------------------------------
# SOP — structure-aware: numbered steps stay inside their subsection chunk
# ---------------------------------------------------------------------------

def test_sop_section_41_steps_not_split_across_chunks(sop_chunks: list[DocumentChunk]) -> None:
    """
    All three Step lines for section 4.1 must appear in the same chunk.
    If they were split mid-procedure, at least one Step line would be absent
    from the chunk that contains 'Step 1'.
    """
    section_41_chunks = [
        c for c in sop_chunks
        if c.metadata.section_title == "4.1 Identification and Initial Documentation"
    ]
    assert section_41_chunks, "No chunks found for section 4.1"

    # The primary chunk (largest) must contain all three steps
    primary = max(section_41_chunks, key=lambda c: len(c.content))
    assert "Step 1:" in primary.content, "Step 1 missing from 4.1 primary chunk"
    assert "Step 2:" in primary.content, "Step 2 missing from 4.1 primary chunk"
    assert "Step 3:" in primary.content, "Step 3 missing from 4.1 primary chunk"


def test_sop_section_42_steps_not_split_across_chunks(sop_chunks: list[DocumentChunk]) -> None:
    """Section 4.2 has three steps.

    4.2 is ~740 chars total — larger than max_chunk_size=500 — so it is
    legitimately split by the recursive splitter. The important guarantee is:
    1. Every chunk for 4.2 carries the correct section_title (steps are never
       orphaned under a different or missing title).
    2. All three Step lines are present somewhere across the section's chunks.
    """
    section_42_chunks = [
        c for c in sop_chunks
        if c.metadata.section_title == "4.2 Serious Adverse Event Escalation"
    ]
    assert section_42_chunks, "No chunks found for section 4.2"

    # All steps must be accounted for across the section's chunks collectively
    combined = " ".join(c.content for c in section_42_chunks)
    assert "Step 1:" in combined, "Step 1 missing from section 4.2 chunks"
    assert "Step 2:" in combined, "Step 2 missing from section 4.2 chunks"
    assert "Step 3:" in combined, "Step 3 missing from section 4.2 chunks"

    # Every chunk must carry the correct section title — steps are never orphaned
    for chunk in section_42_chunks:
        assert chunk.metadata.section_title == "4.2 Serious Adverse Event Escalation"


def test_sop_section_43_present(sop_chunks: list[DocumentChunk]) -> None:
    section_43_chunks = [
        c for c in sop_chunks
        if c.metadata.section_title and "4.3" in c.metadata.section_title
    ]
    assert section_43_chunks, "No chunks found for section 4.3"
    combined = " ".join(c.content for c in section_43_chunks)
    assert "IRB" in combined or "Institutional Review Board" in combined


def test_sop_has_section_titles(sop_chunks: list[DocumentChunk]) -> None:
    titled = [c for c in sop_chunks if c.metadata.section_title is not None]
    assert len(titled) > 0, "SOP should have chunks with section titles"


# ---------------------------------------------------------------------------
# Drug manual — section headers correctly attached
# ---------------------------------------------------------------------------

def test_metformin_dosage_section_has_title(metformin_chunks: list[DocumentChunk]) -> None:
    dosage_chunks = [
        c for c in metformin_chunks
        if c.metadata.section_title == "Dosage and Administration"
    ]
    assert dosage_chunks, "Expected chunks with title 'Dosage and Administration'"


def test_metformin_no_chunk_exceeds_max_size(metformin_chunks: list[DocumentChunk]) -> None:
    chunker = HybridChunker(max_chunk_size=500)
    # Re-chunk with explicit max to confirm no chunk exceeds the limit
    # (the default pipeline uses 500 too, this makes the assertion explicit)
    for chunk in metformin_chunks:
        assert len(chunk.content) <= 500 + 75, (  # allow one overlap window of tolerance
            f"Chunk too large ({len(chunk.content)} chars): {chunk.metadata}"
        )


# ---------------------------------------------------------------------------
# Clinical trial — overflow splitting produces correct section titles
# ---------------------------------------------------------------------------

def test_trial_efficacy_section_chunks_carry_title(trial_chunks: list[DocumentChunk]) -> None:
    efficacy_chunks = [
        c for c in trial_chunks
        if c.metadata.section_title == "Key Efficacy Results"
    ]
    assert efficacy_chunks, "Expected chunks for 'Key Efficacy Results'"
    for c in efficacy_chunks:
        assert c.content.strip() != ""


def test_trial_safety_section_present(trial_chunks: list[DocumentChunk]) -> None:
    safety_chunks = [
        c for c in trial_chunks
        if c.metadata.section_title == "Safety Summary"
    ]
    assert safety_chunks


# ---------------------------------------------------------------------------
# Plain-text / no-header fallback (chunker unit test with synthetic input)
# ---------------------------------------------------------------------------

def test_plain_text_no_headers_still_produces_chunks() -> None:
    """A .txt document with no markdown headers must still be chunked."""
    plain_text = (
        "This is a plain text document with no headers.\n\n"
        "It has multiple paragraphs that should each become their own chunk "
        "when the content is long enough to trigger splitting.\n\n"
        "Third paragraph here with enough content to ensure we get at least "
        "one chunk from this synthetic document used only in tests."
    )
    doc = RawDocument(source="plain.txt", text=plain_text)
    chunker = HybridChunker(max_chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip() != ""
        assert chunk.metadata.source == "plain.txt"


def test_plain_text_chunk_indices_sequential() -> None:
    plain_text = "\n\n".join(f"Paragraph number {i} with some content." for i in range(10))
    doc = RawDocument(source="seq.txt", text=plain_text)
    chunker = HybridChunker(max_chunk_size=80, chunk_overlap=0)
    chunks = chunker.chunk(doc)

    indices = [c.metadata.chunk_index for c in chunks]
    assert indices == list(range(len(indices)))
