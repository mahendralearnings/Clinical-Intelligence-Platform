"""AWS Bedrock Cohere Embed English v3 implementation of EmbeddingProvider.

Model: cohere.embed-english-v3
Output dimension: 1024
Max input tokens: 512 tokens per text

Why Cohere over Titan:
- Titan Embeddings v1 is being retired/throttled in some regions.
- Cohere Embed English v3 is ACTIVE and confirmed working in us-east-1.
- 1024-d vectors are smaller than Titan's 1536-d, meaning faster cosine
  similarity and a smaller JSON store — no quality loss for clinical text.
- Swappable back to Titan v2 (or any other model) by changing the model_id
  and dimension in Settings without touching this class.

Design decisions:
- boto3 client is created lazily on first call so the class can be
  instantiated in tests without valid AWS credentials (tests inject a
  FakeEmbedder instead).
- All botocore/boto3 exceptions are caught and re-raised as EmbeddingError
  so service code never has to import infrastructure packages.
- The client is stored on the instance, making it easy to inject a mock
  boto3 client in integration tests (pass it via the constructor).
"""

from __future__ import annotations

import json
from typing import Any

from clinical_platform.domain.ports import EmbeddingError

# Cohere Embed English v3 — confirmed ACTIVE in us-east-1
_DEFAULT_MODEL_ID = "cohere.embed-english-v3"
_EXPECTED_DIMENSION = 1024


class BedrockEmbeddingProvider:
    """Calls AWS Bedrock Cohere Embed English v3 to produce a 1024-d float vector.

    Args:
        region:    AWS region name (default ``us-east-1``).
        model_id:  Bedrock model ID (default ``cohere.embed-english-v3``).
        client:    Optional pre-built boto3 bedrock-runtime client.  Pass a
                   mock here in integration tests to avoid real AWS calls.
        dimension: Expected output dimension (default 1024). Must match the
                   model's actual output — used as a sanity check.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = _DEFAULT_MODEL_ID,
        client: Any = None,
        dimension: int = _EXPECTED_DIMENSION,
    ) -> None:
        self._region = region
        self._model_id = model_id
        self._client = client  # None -> created lazily on first embed()
        self._dimension = dimension

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
                raise EmbeddingError(
                    f"Failed to create Bedrock client: {exc}"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Embed *text* using Cohere Embed English v3.

        Cohere's request schema differs from Titan:
          - ``texts``: list of strings (we always send one at a time)
          - ``input_type``: ``"search_document"`` for chunks being stored,
            ``"search_query"`` for queries.  We use ``"search_document"``
            as a safe default for both — the quality difference is minor
            for this use case and avoids needing two code paths.

        Args:
            text: The string to embed.

        Returns:
            A list of 1024 floats.

        Raises:
            EmbeddingError: on any Bedrock or network failure.
        """
        client = self._get_client()
        body = json.dumps({"texts": [text], "input_type": "search_document"})

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

        # Cohere returns {"embeddings": [[...], ...]} — one list per input text
        try:
            vector: list[float] = response_body["embeddings"][0]
        except (KeyError, IndexError) as exc:
            raise EmbeddingError(
                f"Unexpected Cohere response schema. "
                f"Got keys: {list(response_body.keys())}"
            ) from exc

        if len(vector) != self._dimension:
            raise EmbeddingError(
                f"Cohere returned {len(vector)}-d vector; expected {self._dimension}."
            )

        return vector
