"""
Lightweight, rule-based guardrails - no AI calls needed. This is
deliberately the "first line of defense" - simple pattern matching,
not a full ML-based safety classifier. Real production systems often
layer this with AI-based checks too, but that's a heavier, separate
addition, not needed to demonstrate the core concept.
"""

import re

# A short, illustrative list - not exhaustive. Real systems use much
# larger, maintained pattern lists or a dedicated library for this.
_INJECTION_PATTERNS = [
    r"ignore (all |your )?(previous|prior|above) instructions",
    r"disregard (all |your )?(previous|prior|above) instructions",
    r"you are now",
    r"system prompt",
    r"reveal your (system )?prompt",
    r"act as (if you are|a) (?!.*(doctor|researcher|clinician))",
]

_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


class PromptInjectionDetected(Exception):
    """Raised when input matches a known prompt-injection pattern."""

    def __init__(self, matched_pattern: str) -> None:
        super().__init__(f"Potential prompt injection detected: pattern '{matched_pattern}'")
        self.matched_pattern = matched_pattern


def check_for_prompt_injection(text: str) -> None:
    """Raises PromptInjectionDetected if the text matches a known attack pattern."""
    lowered = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            raise PromptInjectionDetected(matched_pattern=pattern)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Returns (redacted_text, list_of_pii_types_found). Replaces detected
    PII with a placeholder like [REDACTED_EMAIL] - never logs or passes
    the real value onward.
    """
    redacted = text
    found_types: list[str] = []

    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(redacted):
            found_types.append(pii_type)
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)

    return redacted, found_types