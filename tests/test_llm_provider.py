"""Tests for the LLM Gateway — Phase 4 (Piece 2 + Piece 3).

Unit tests run without AWS credentials.
Retry tests use wait_none() so no real sleeps occur.
The optional integration test (BEDROCK_INTEGRATION=1) calls real Claude.
"""

from __future__ import annotations

import os

import pytest
from tenacity import wait_none

from clinical_platform.domain.ports import LLMError, LLMProvider
from clinical_platform.infrastructure.llm_providers.bedrock_llm_provider import (
    BedrockLLMProvider,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeLLMProvider:
    """LLMProvider test double — returns a fixed string, makes no AWS calls.

    Satisfies the LLMProvider Protocol structurally (no inheritance needed).
    Accepts the extended generate() signature and ignores the new params.
    """

    def __init__(self, response: str = "fake response") -> None:
        self._response = response

    def generate(
        self,
        prompt: str,  # noqa: ARG002
        *,
        system_prompt: str | None = None,  # noqa: ARG002
        max_tokens: int = 512,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
    ) -> str:
        return self._response


class _FakeBedrockClient:
    """Minimal boto3 bedrock-runtime client fake for retry tests.

    Raises a simulated error for the first *fail_times* converse() calls,
    then returns *success_response* on subsequent calls.
    """

    def __init__(
        self,
        fail_times: int,
        error_message: str = "ThrottlingException",
        success_response: dict | None = None,
    ) -> None:
        self._fail_times = fail_times
        self._error_message = error_message
        self._success = success_response or {
            "output": {
                "message": {
                    "content": [{"text": "success after retry"}]
                }
            }
        }
        self.call_count = 0

    def converse(self, **_kwargs: object) -> dict:  # type: ignore[return]
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise RuntimeError(self._error_message)
        return self._success


# ---------------------------------------------------------------------------
# Protocol conformance — Piece 2 tests (still passing)
# ---------------------------------------------------------------------------


def test_fake_provider_satisfies_protocol() -> None:
    fake = FakeLLMProvider()
    assert isinstance(fake, LLMProvider)


def test_fake_provider_returns_non_empty_string() -> None:
    fake = FakeLLMProvider()
    result = fake.generate("any prompt")
    assert isinstance(result, str)
    assert len(result) > 0


def test_fake_provider_returns_configured_response() -> None:
    expected = "This is a custom clinical AI response."
    fake = FakeLLMProvider(response=expected)
    assert fake.generate("what is metformin?") == expected
    assert fake.generate("completely different prompt") == expected


def test_fake_provider_default_response_is_fake_response() -> None:
    fake = FakeLLMProvider()
    assert fake.generate("hello") == "fake response"


# ---------------------------------------------------------------------------
# Extended signature — Piece 3 backward-compatibility
# ---------------------------------------------------------------------------


def test_fake_provider_accepts_system_prompt_and_max_tokens() -> None:
    """New keyword params must be accepted without error."""
    fake = FakeLLMProvider(response="ok")
    result = fake.generate(
        "summarise this",
        system_prompt="You are a clinical assistant.",
        max_tokens=256,
    )
    assert result == "ok"


# ---------------------------------------------------------------------------
# Retry behaviour tests — Piece 3 (no sleeps via wait_none())
# ---------------------------------------------------------------------------


def test_retries_on_throttling_then_succeeds() -> None:
    """Provider retries twice on ThrottlingException and returns the eventual
    success response on the third attempt."""
    fake_client = _FakeBedrockClient(fail_times=2)
    provider = BedrockLLMProvider(
        client=fake_client,
        _wait_strategy=wait_none(),
    )

    result = provider.generate("test prompt")

    assert result == "success after retry"
    assert fake_client.call_count == 3  # 2 failures + 1 success


def test_gives_up_after_max_retries() -> None:
    """Provider raises LLMError after exhausting all 3 attempts."""
    fake_client = _FakeBedrockClient(fail_times=99)  # always fails
    provider = BedrockLLMProvider(
        client=fake_client,
        _wait_strategy=wait_none(),
    )

    with pytest.raises(LLMError, match="3 attempts"):
        provider.generate("test prompt")

    assert fake_client.call_count == 3


def test_no_retry_on_non_retryable_error() -> None:
    """AccessDeniedException is not retried — fails immediately after 1 call."""
    fake_client = _FakeBedrockClient(
        fail_times=99,
        error_message="AccessDeniedException: User not authorized",
    )
    provider = BedrockLLMProvider(
        client=fake_client,
        _wait_strategy=wait_none(),
    )

    with pytest.raises(LLMError):
        provider.generate("test prompt")

    assert fake_client.call_count == 1


def test_system_prompt_and_max_tokens_forwarded() -> None:
    """system_prompt and max_tokens are included in the converse() request."""

    class _CapturingClient:
        """Captures the kwargs passed to converse()."""

        def __init__(self) -> None:
            self.last_kwargs: dict = {}

        def converse(self, **kwargs: object) -> dict:
            self.last_kwargs = dict(kwargs)
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    capturing = _CapturingClient()
    provider = BedrockLLMProvider(client=capturing, _wait_strategy=wait_none())
    provider.generate(
        "hello",
        system_prompt="You are a clinical assistant.",
        max_tokens=256,
    )

    assert capturing.last_kwargs["inferenceConfig"] == {"maxTokens": 256, "temperature": 0.0}
    assert capturing.last_kwargs["system"] == [{"text": "You are a clinical assistant."}]


def test_no_system_key_when_system_prompt_is_none() -> None:
    """When system_prompt is None the 'system' key must be absent from the request."""

    class _CapturingClient:
        def __init__(self) -> None:
            self.last_kwargs: dict = {}

        def converse(self, **kwargs: object) -> dict:
            self.last_kwargs = dict(kwargs)
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    capturing = _CapturingClient()
    provider = BedrockLLMProvider(client=capturing, _wait_strategy=wait_none())
    provider.generate("hello")

    assert "system" not in capturing.last_kwargs


# ---------------------------------------------------------------------------
# Optional real-Bedrock integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("BEDROCK_INTEGRATION") != "1",
    reason="Set BEDROCK_INTEGRATION=1 to run real AWS Bedrock calls",
)
def test_bedrock_claude_returns_non_empty_string() -> None:  # pragma: no cover
    """End-to-end: real Claude 3 Haiku via Bedrock.

    Prerequisites:
        - Valid AWS credentials in environment / ~/.aws/credentials
        - Claude 3 Haiku enabled under Bedrock Model Access in us-east-1
        - BEDROCK_INTEGRATION=1 environment variable set
    """
    provider = BedrockLLMProvider()
    result = provider.generate(
        "Say hello in exactly 5 words.",
        system_prompt="You are a helpful assistant.",
        max_tokens=50,
    )
    assert isinstance(result, str)
    assert len(result) > 0
