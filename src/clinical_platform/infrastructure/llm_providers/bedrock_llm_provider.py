"""AWS Bedrock Claude implementation of LLMProvider — Phase 4 Piece 3.

Model:   anthropic.claude-3-haiku-20240307-v1:0
API:     boto3 converse()

Changes from Piece 2:
- Retry logic via tenacity: up to 3 attempts with exponential backoff on
  throttling / transient errors; immediate failure on non-retryable errors.
- Extended generate() signature: system_prompt and max_tokens params.
- _wait_strategy constructor parameter for testable retries (no real sleep
  needed in tests — pass wait_none() instead).
- model_id and region still constructor params; they are now also driven
  from Settings via get_llm_provider() in api/middleware/llm_dependencies.py.
"""

from __future__ import annotations

from typing import Any

from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from clinical_platform.domain.ports import LLMError

_DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
_MAX_ATTEMPTS = 3

# Module-level default wait strategy — avoids B008 (function call in default arg)
_DEFAULT_WAIT = wait_exponential(multiplier=0.5, min=0.5, max=4)

# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

# Error code strings that warrant a retry
_RETRYABLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "RequestTimeout"}
# Error code strings that should never be retried
_NON_RETRYABLE_CODES = {"AccessDeniedException", "ValidationException"}


def _error_code(exc: BaseException) -> str:
    """Extract the AWS error code from a botocore ClientError, or return ''."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return ""


def _is_retryable(exc: BaseException) -> bool:
    """Return True if *exc* represents a transient error worth retrying."""
    code = _error_code(exc)
    if code in _NON_RETRYABLE_CODES:
        return False
    if code in _RETRYABLE_CODES:
        return True
    # Fallback: check the string representation for known retryable keywords
    msg = str(exc)
    if any(k in msg for k in _NON_RETRYABLE_CODES):
        return False
    return any(k in msg for k in _RETRYABLE_CODES)


# ---------------------------------------------------------------------------
# BedrockLLMProvider
# ---------------------------------------------------------------------------


class BedrockLLMProvider:
    """Calls AWS Bedrock Claude via converse() with retry and configurable params.

    Args:
        region:          AWS region name (default ``us-east-1``).
        model_id:        Bedrock model ID (default Claude 3 Haiku).
        client:          Optional pre-built boto3 bedrock-runtime client.
                         Pass a mock/fake here in tests.
        _wait_strategy:  tenacity wait strategy. Default is exponential backoff.
                         Pass ``wait_none()`` in tests to skip real sleeps while
                         still exercising the full retry code path.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = _DEFAULT_MODEL_ID,
        client: Any = None,
        _wait_strategy: Any = _DEFAULT_WAIT,
    ) -> None:
        self._region = region
        self._model_id = model_id
        self._client = client
        self._wait_strategy = _wait_strategy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-untyped]  # noqa: PLC0415

                self._client = boto3.client(
                    "bedrock-runtime", region_name=self._region
                )
            except Exception as exc:
                raise LLMError(f"Failed to create Bedrock client: {exc}") from exc
        return self._client

    def _single_call(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """One attempt at calling converse() — no retry logic here.

        Raises the raw boto3 exception so tenacity can inspect it.
        """
        client = self._get_client()

        request: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system_prompt:
            request["system"] = [{"text": system_prompt}]

        result = client.converse(**request)

        try:
            return result["output"]["message"]["content"][0]["text"]  # type: ignore[no-any-return]
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Unexpected response structure from '{self._model_id}'. "
                f"Output keys: {list(result.get('output', {}).keys())}"
            ) from exc

    # ------------------------------------------------------------------
    # LLMProvider protocol
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Send *prompt* to Claude and return the text reply.

        Automatically retries up to 3 times on throttling / transient errors
        with exponential backoff. Fails immediately on non-retryable errors
        (e.g. AccessDeniedException, ValidationException).

        Args:
            prompt:        User message.
            system_prompt: Optional system-level instruction.
            max_tokens:    Maximum tokens in the response (default 512).

        Returns:
            The model's text response as a plain string.

        Raises:
            LLMError: after exhausting retries or on non-retryable failure.
        """
        # Build the retrying caller inline so we can inject self._wait_strategy
        @retry(
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=self._wait_strategy,
            retry=retry_if_exception(_is_retryable),
            reraise=False,  # we catch RetryError below and wrap as LLMError
        )
        def _call_with_retry() -> str:
            return self._single_call(prompt, system_prompt, max_tokens, temperature)

        try:
            return _call_with_retry()
        except RetryError as exc:
            # tenacity wraps the last exception inside RetryError
            last_exc = exc.last_attempt.exception()
            last: BaseException = last_exc if last_exc is not None else exc
            raise LLMError(
                f"Bedrock converse() failed after {_MAX_ATTEMPTS} attempts "
                f"for model '{self._model_id}': {last}"
            ) from last
        except LLMError:
            raise  # response-parsing errors from _single_call — pass through
        except Exception as exc:
            # Non-retryable boto3 error — _is_retryable returned False so
            # tenacity re-raised it directly (reraise=False only suppresses
            # RetryError, not the original exception when retry=False).
            raise LLMError(
                f"Bedrock converse() failed for model '{self._model_id}': {exc}"
            ) from exc
