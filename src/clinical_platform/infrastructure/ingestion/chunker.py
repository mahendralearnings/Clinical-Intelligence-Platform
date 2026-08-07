"""Hybrid chunker: markdown-header split first, recursive split for overflow."""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from clinical_platform.domain.chunk import ChunkMetadata, DocumentChunk
from clinical_platform.domain.document import RawDocument

# Headers the first-pass splitter recognises, in order of priority.
_HEADERS_TO_SPLIT_ON = [("##", "H2"), ("###", "H3")]


class HybridChunker:
    def __init__(self, max_chunk_size: int = 500, chunk_overlap: int = 75) -> None:
        self._max_chunk_size = max_chunk_size
        self._chunk_overlap = chunk_overlap

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
            strip_headers=False,  # keep header text inside the section body
        )
        self._rc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, doc: RawDocument) -> list[DocumentChunk]:
        # ------------------------------------------------------------------
        # Pass 1 — split on ## / ### headers
        # ------------------------------------------------------------------
        sections = self._header_splitter.split_text(doc.text)

        raw_chunks: list[tuple[str, str | None]] = []  # (content, section_title)

        for section in sections:
            # Prefer the most specific header (H3 > H2), fall back to None
            section_title: str | None = (
                section.metadata.get("H3")
                or section.metadata.get("H2")
                or None
            )
            content = section.page_content.strip()

            if not content:
                continue

            # ------------------------------------------------------------------
            # Pass 2 — overflow split for sections that are still too large
            # ------------------------------------------------------------------
            if len(content) > self._max_chunk_size:
                sub_texts = self._rc_splitter.split_text(content)
                for sub in sub_texts:
                    sub = sub.strip()
                    if sub:
                        raw_chunks.append((sub, section_title))
            else:
                raw_chunks.append((content, section_title))

        # ------------------------------------------------------------------
        # Assemble final DocumentChunk objects with sequential index
        # ------------------------------------------------------------------
        return [
            DocumentChunk(
                content=content,
                metadata=ChunkMetadata(
                    source=doc.source,
                    section_title=section_title,
                    chunk_index=idx,
                ),
            )
            for idx, (content, section_title) in enumerate(raw_chunks)
        ]
