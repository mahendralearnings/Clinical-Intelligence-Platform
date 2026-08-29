# Phase 4 — LLM Gateway: Requirements (Piece 3 — Production Hardening)

## Context

Piece 2 established the `LLMProvider` protocol and a minimal
`BedrockLLMProvider` that makes a single `converse()` call with no
resilience. Piece 3 makes it production-ready: retry logic, configurable
parameters, and a FastAPI dependency so Phase 5 can wire it in with
`Depends(get_llm_provider)`.

---

## Functional Requirements

### FR-1 Retry logic on throttling / transient errors
`BedrockLLMProvider.generate()` shall automatically retry up to **3
attempts** (1 initial + 2 retries) when the underlying `converse()` call
raises a throttling or transient error. Between attempts it shall wait
with **exponential backoff**: 0.5 s after the first failure, 1.0 s after
the second.

- On a **non-retryable** error (e.g. `AccessDeniedException`,
  `ValidationException`) it shall fail immediately — do not retry.
- After exhausting all attempts it shall raise `LLMError` with a clear
  message including the attempt count and the underlying error.
- Implementation uses `tenacity` (new dependency).

### FR-2 Extended `generate()` signature
`generate()` gains two optional keyword parameters that do not break
existing callers:

```python
def generate(
    self,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 512,
) -> str:
```

- `system_prompt`: if provided, passed as the `system` field in the
  `converse()` request.
- `max_tokens`: forwarded as `inferenceConfig={"maxTokens": max_tokens}`.

### FR-3 Extended `LLMProvider` Protocol signature
`domain/ports.py` — update `LLMProvider.generate()` to match the new
signature (keyword-only defaults mean existing callers are unaffected).

### FR-4 Settings fields for LLM
`core/config.py` (`Settings`) gains two new fields:

| Field | Type | Default |
|-------|------|---------|
| `llm_model_id` | `str` | `"anthropic.claude-3-haiku-20240307-v1:0"` |
| `llm_region` | `str` | `"us-east-1"` |

Both are independently overridable via environment variable / `.env`.
`llm_region` is kept separate from `bedrock_region` (used by embeddings)
so each can point to a different region if needed.

### FR-5 `get_llm_provider()` FastAPI dependency
A new file `api/middleware/llm_dependencies.py`:

```python
def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMProvider:
```

Returns a `BedrockLLMProvider` constructed from `Settings`. Same pattern
as `get_auth_service()` and `get_retrieval_service()`.

### FR-6 Updated `FakeLLMProvider`
`FakeLLMProvider` gains the new keyword parameters (accepted and ignored)
so existing tests continue to pass without modification.

### FR-7 Retry behaviour tests (no AWS calls, no real sleeps)
`tests/test_llm_provider.py` gains:

- `test_retries_on_throttling_then_succeeds` — fake client fails twice
  with throttling, succeeds on the third call. Assert result is correct
  and exactly 3 `converse()` calls were made.
- `test_gives_up_after_max_retries` — fake client always raises
  throttling. Assert `LLMError` is raised.
- `test_no_retry_on_non_retryable_error` — fake client raises
  `AccessDeniedException`. Assert `LLMError` is raised after exactly
  1 call (no retries).

Tests use `wait_none()` injected via `_wait_strategy` constructor
parameter — no actual sleeping during tests.

---

## Out of Scope (Piece 3)

- Streaming / async generation (future phase)
- Token counting or cost tracking (Phase 5+)
- Wiring into any API route (Phase 5)
- Multiple model support / model selection logic at request time
