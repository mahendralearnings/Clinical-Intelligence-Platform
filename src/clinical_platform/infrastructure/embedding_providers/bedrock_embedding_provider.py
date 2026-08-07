"""AWS Bedrock Titan Embeddings implementation of EmbeddingProvider.

Model: amazon.titan-embed-text-v1
Output dimension: 1536
Max input tokens: 8192

Design decisions:
- boto3 client is created lazily on first call so the class can be
  instantiated in tests without valid AWS credentials (tests inject a
  FakeEmbedder instead, but it's still nice not to fail at import time).
- All botocore/boto3 exceptions are caught and re-raised as EmbeddingError
  so service code never has to import infrastructure packages.
- The client is stored on the instance, making it easy to inject a mock
  boto3 client in integration tests (pass it via the constructor).
"""

from __future__ import annotations

import json
from typing import Any

from clinical_platform.domain.ports import EmbeddingError

# Titan Embeddings G1 – Text
_DEFAULT_MODEL_ID = "amazon.titan-embed-text-v1"
_EXPECTED_DIMENSION = 1536


class BedrockEmbeddingProvider:
    """Calls AWS Bedrock Titan Embeddings to produce a 1536-d float vector.

    Args:
        region:   AWS region name (default ``us-east-1``).
        model_id: Bedrock model ID (default ``amazon.titan-embed-text-v1``).
        client:   Optional pre-built boto3 bedrock-runtime client.  Pass a
                  mock here in integration tests to avoid real AWS calls.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = _DEFAULT_MODEL_ID,
        client: Any = None,
    ) -> None:
        self._region = region
        self._model_id = model_id
        self._client = client  # None -> created lazily on first embed()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-not-found]  # noqa: PLC0415

                self._client = boto3.client(
                    "bedrock-runtime", region_name=self._region
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to create Bedrock client: {exc}"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed *text* using Titan Embeddings G1.

        Args:
            text: The string to embed (<= 8192 tokens).

        Returns:
            A list of 1536 floats.

        Raises:
            EmbeddingError: on any Bedrock or network failure.
        """
        client = self._get_client()
        body = json.dumps({"inputText": text})

        try:
            response = client.invoke_model(
                modelId=self._model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read())
        except Exception as exc:
            raise EmbeddingError(
                f"Bedrock invoke_model failed for model '{self._model_id}': {exc}"
            ) from exc

        try:
            vector: list[float] = response_body["embedding"]
        except KeyError as exc:
            raise EmbeddingError(
                f"Unexpected Bedrock response schema — 'embedding' key missing. "
                f"Got keys: {list(response_body.keys())}"
            ) from exc

        if len(vector) != _EXPECTED_DIMENSION:
            raise EmbeddingError(
                f"Titan returned {len(vector)}-d vector; expected {_EXPECTED_DIMENSION}."
            )

        return vector
