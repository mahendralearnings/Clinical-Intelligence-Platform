"""Domain model for a raw loaded document.

Pure Python — no FastAPI, no LangChain, no I/O.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RawDocument:
    """A document as read from disk, before any splitting."""

    source: str  # filename only, e.g. "sop_adverse_event_reporting.md"
    text: str    # full raw file content
