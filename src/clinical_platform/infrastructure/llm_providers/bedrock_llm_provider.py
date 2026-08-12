"""AWS Bedrock Claude implementation of LLMProvider.

Model:   anthropic.claude-3-haiku-20240307-v1:0
API:     boto3 converse() — proven working in scripts/test_bedrock_connection.py

Design decisions:
- boto3 client created lazily on first generate() call (same pattern as
  BedrockEmbeddingProvider) so the class is safe to instantiate in any
  context without requiring live AWS credentials.
- All boto3/botocore exceptions are caught and re-raised as LLMError so
  service code never has to import infrastructure packages.
- model_id and region are constructor parameters (not yet in Settings —
  that wiring happens in Piece 3 when the provider is exposed as a
  FastAPI dependency).
- No retries, no streaming, no system prompt yet — Piece 3 concern.
"""

from __future__ import annotations

from typing import Any

from clinical_platform.domain.ports import LLMError

_DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"


class BedrockLLMProvider:
    """Calls AWS Bedrock Claude via converse() and returns the text reply.

    Args:
        region:   AWS region name (default ``us-east-1``).
        model_id: Bedrock model ID (default Claude 3 Haiku).
        client:   Optional pre-built boto3 bedrock-runtime client.
                  Pass a mock here in integration tests.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = _DEFAULT_MODEL_ID,
        client: Any = None,
    ) -> None:
        self._region = region
        self._model_id = model_id
        self._client = client  # None -> created lazily on first generate()

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

    # ------------------------------------------------------------------
    # LLMProvider protocol
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send *prompt* to Claude and return the text reply.

        Args:
            prompt: Plain-string user message.

        Returns:
            The model's text response as a plain string.

        Raises:
            LLMError: on any Bedrock or network failure.
        """
        client = self._get_client()

        try:
            result = client.converse(
                modelId=self._model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            )
        except Exception as exc:
            raise LLMError(
                f"Bedrock converse() failed for model '{self._model_id}': {exc}"
            ) from exc

        try:
            return result["output"]["message"]["content"][0]["text"]  # type: ignore[no-any-return]
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Unexpected response structure from '{self._model_id}'. "
                f"Output keys: {list(result.get('output', {}).keys())}"
            ) from exc
