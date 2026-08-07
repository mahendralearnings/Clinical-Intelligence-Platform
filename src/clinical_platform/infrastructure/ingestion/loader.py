"""Loads .md and .txt files from a directory into RawDocument objects."""

from pathlib import Path

from clinical_platform.domain.document import RawDocument

SUPPORTED_EXTENSIONS = {".md", ".txt"}


class DocumentLoader:
    def __init__(self, docs_dir: Path) -> None:
        self._docs_dir = docs_dir

    def load(self) -> list[RawDocument]:
        """Read every supported file in docs_dir; skip everything else."""
        documents: list[RawDocument] = []

        for path in sorted(self._docs_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue  # skips .gitkeep, images, etc.

            text = path.read_text(encoding="utf-8")
            documents.append(RawDocument(source=path.name, text=text))

        return documents
