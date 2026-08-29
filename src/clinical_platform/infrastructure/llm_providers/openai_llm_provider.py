"""
Direct OpenAI API LLM provider - another simple alternative when 
Bedrock/Anthropic are unavailable.
"""

import openai

from clinical_platform.domain.ports import LLMError

from langsmith import traceable


class OpenAILLMProvider:
    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._client: openai.OpenAI | None = None

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    @traceable(name="llm_generate")

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
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self._model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"OpenAI API call failed: {exc}") from exc