"""
Chunking strategy comparison — a throwaway experiment, NOT part of the
real application. Run this to SEE how different strategies split our
3 sample documents differently, before deciding what to build for real.
"""

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

SAMPLE_DOCS_DIR = Path("data/sample_documents")


def load_doc(filename: str) -> str:
    return (SAMPLE_DOCS_DIR / filename).read_text(encoding="utf-8")


def show_chunks(label: str, chunks: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print(f"{label} — {len(chunks)} chunks")
    print("=" * 70)
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.strip().replace("\n", " ")[:100]
        print(f"  [{i}] ({len(chunk)} chars) {preview}...")


def run_recursive_splitter(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
    """Strategy: fixed-size-ish, but respects paragraph/sentence boundaries when possible."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def run_structure_aware_splitter(text: str) -> list[str]:
    """Strategy: split on markdown headers first (## and ###), keeping each
    section together as one chunk — then we could recursively split any
    section that's still too big (not shown here, kept simple for comparison)."""
    headers_to_split_on = [("##", "section"), ("###", "subsection")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(text)
    return [doc.page_content for doc in docs]


def compare_on_document(filename: str) -> None:
    text = load_doc(filename)
    print(f"\n\n{'#' * 70}")
    print(f"# DOCUMENT: {filename}  ({len(text)} total characters)")
    print("#" * 70)

    recursive_chunks = run_recursive_splitter(text)
    show_chunks("STRATEGY: Recursive Character Splitting", recursive_chunks)

    structure_chunks = run_structure_aware_splitter(text)
    show_chunks("STRATEGY: Structure-Aware (Markdown Headers)", structure_chunks)


if __name__ == "__main__":
    compare_on_document("sop_adverse_event_reporting.md")
    compare_on_document("drug_manual_metformin.md")
    compare_on_document("clinical_trial_summary_204.md")