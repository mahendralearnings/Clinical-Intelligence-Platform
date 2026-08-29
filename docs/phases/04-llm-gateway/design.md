# Phase 4 — LLM Gateway: Design (Piece 2 — Core Interface)

## 1. Scope

Exactly three things change in the codebase:

| Change | File |
|--------|------|
| Append `LLMProvider` Protocol + `LLMError` exception | `domain/ports.py` |
| New Bedrock Claude implementation | `infrastructure/llm_providers/bedrock_llm_provider.py` |
| New test file | `tests/test_llm_provider.py` |

`main.py`, all existing routes, `config.py`, and every Phase 3 file are
**untouched**.

---

## 2. Domain Layer Addition

### New items appended to `domain/ports.py`

```python
class LLMError(Exception):
    """Raised when an LLM provider fails to generate a response."""

@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
```

**Why append to `ports.py` rather than a new file?**
`ports.py` is already the home for all domain-level interfaces and
exceptions (`EmbeddingError`, `DimensionMismatchError`, `EmbeddingProvider`,
`VectorStore`). Adding `LLMProvider` here keeps the "one place to look for
all swappable interfaces" rule intact. The file stays small (~15 new lines).

**Why only `generate(prompt: str) -> str`?**
The simplest signature that unblocks Phase 5's RAG query service. System
prompts, temperature, max_tokens, and message history are Piece 3 additions
— they extend this signature rather than replace it, so nothing built here
is thrown away.

---

## 3. Infrastructure Adapter

### `infrastructure/llm_providers/bedrock_llm_provider.py`

```python
class BedrockLLMProvider:
    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        client: Any = None,
    ) -> None: ...

    def generate(self, prompt: str) -> str: ...
```

**Request construction** — mirrors `test_bedrock_connection.py` exactly:

```python
result = client.converse(
    modelId=self._model_id,
    messages=[{"role": "user", "content": [{"text": prompt}]}],
)
return result["output"]["message"]["content"][0]["text"]
```

**Lazy client** — `self._client` starts as `None`, created on first
`generate()` call. Same pattern as `BedrockEmbeddingProvider`.

**Error wrapping** — any exception from `converse()` is caught and
re-raised as `LLMError`. Covers:
- `NoCredentialsError` (no AWS config)
- `AccessDeniedException` (model not enabled)
- `ThrottlingException` (rate limit)
- `ValidationException` (bad prompt format)

No retry logic — that is Piece 3.

**Model ID as constructor default** — not in `Settings` yet. Piece 3 will
add `llm_model_id` and `llm_region` to `config.py` when the provider gets
wired into a FastAPI dependency.

---

## 4. Test Double

### `FakeLLMProvider` (inside `tests/test_llm_provider.py`)

```python
class FakeLLMProvider:
    def __init__(self, response: str = "fake response") -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response
```

Satisfies `LLMProvider` structurally — no inheritance needed. The `prompt`
parameter is accepted and ignored, which is correct: test doubles should
have the right signature but need not do real work unless a test specifically
requires it.

---

## 5. File Changeset

### New files
```
docs/phases/04-llm-gateway/requirements.md
docs/phases/04-llm-gateway/design.md
src/clinical_platform/infrastructure/llm_providers/bedrock_llm_provider.py
tests/test_llm_provider.py
```

### Modified files
```
src/clinical_platform/domain/ports.py   ← append LLMError + LLMProvider
```

### Untouched
```
Everything else — main.py, config.py, all Phase 3 files, all other tests
```

---

## 6. What Piece 3 Adds (preview)

- `system_prompt: str` and `max_tokens: int` parameters to `generate()`
- Retry logic with exponential backoff (`tenacity`)
- `config.py` gets `llm_model_id` and `llm_region` settings
- `get_llm_provider()` FastAPI dependency ready for Phase 5
