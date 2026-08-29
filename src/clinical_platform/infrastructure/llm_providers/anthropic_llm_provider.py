"""
Direct Anthropic API LLM provider - simple alternative to Bedrock.
Used when AWS Bedrock is rate-limited or unavailable.
"""

import anthropic

from clinical_platform.domain.ports import LLMError


class AnthropicLLMProvider:
    def __init__(
        self,
        api_key: str,
        model_id: str = "claude-3-5-haiku-20241022",
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self._model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as exc:
            raise LLMError(f"Anthropic API call failed: {exc}") from exc