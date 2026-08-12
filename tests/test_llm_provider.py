"""Tests for the LLM Gateway — Phase 4 Piece 2.

All unit tests run without AWS credentials using FakeLLMProvider.
The optional integration test (guarded by BEDROCK_INTEGRATION=1) calls
the real Claude 3 Haiku model via Bedrock.
"""

import os

import pytest

from clinical_platform.domain.ports import LLMProvider

# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class FakeLLMProvider:
    """LLMProvider test double — returns a fixed string, makes no AWS calls.

    Satisfies the LLMProvider Protocol structurally (no inheritance needed).
    """

    def __init__(self, response: str = "fake response") -> None:
        self._response = response

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return self._response


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


def test_fake_provider_satisfies_protocol() -> None:
    """FakeLLMProvider must satisfy LLMProvider structurally so it can be
    injected anywhere a real provider is expected."""
    fake = FakeLLMProvider()
    assert isinstance(fake, LLMProvider)


# ---------------------------------------------------------------------------
# FakeLLMProvider behaviour tests
# ---------------------------------------------------------------------------


def test_fake_provider_returns_non_empty_string() -> None:
    fake = FakeLLMProvider()
    result = fake.generate("any prompt")
    assert isinstance(result, str)
    assert len(result) > 0


def test_fake_provider_returns_configured_response() -> None:
    """The configured response comes back verbatim regardless of the prompt."""
    expected = "This is a custom clinical AI response."
    fake = FakeLLMProvider(response=expected)
    assert fake.generate("what is metformin?") == expected
    assert fake.generate("completely different prompt") == expected


def test_fake_provider_default_response_is_fake_response() -> None:
    fake = FakeLLMProvider()
    assert fake.generate("hello") == "fake response"


# ---------------------------------------------------------------------------
# Optional real-Bedrock integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("BEDROCK_INTEGRATION") != "1",
    reason="Set BEDROCK_INTEGRATION=1 to run real AWS Bedrock calls",
)
def test_bedrock_claude_returns_non_empty_string() -> None:  # pragma: no cover
    """End-to-end test using the real Claude 3 Haiku model.

    Prerequisites:
        - Valid AWS credentials in environment / ~/.aws/credentials
        - Claude 3 Haiku enabled under Bedrock Model Access in us-east-1
        - BEDROCK_INTEGRATION=1 environment variable set
    """
    from clinical_platform.infrastructure.llm_providers.bedrock_llm_provider import (
        BedrockLLMProvider,
    )

    provider = BedrockLLMProvider()
    result = provider.generate("Say hello in exactly 5 words.")

    assert isinstance(result, str)
    assert len(result) > 0
