# Phase 4 — LLM Gateway: Design (Piece 3 — Production Hardening)

## 1. Scope

| Change | File | Type |
|--------|------|------|
| Update `LLMProvider.generate()` signature | `domain/ports.py` | Modify |
| Add retry + new params to `BedrockLLMProvider` | `infrastructure/llm_providers/bedrock_llm_provider.py` | Rewrite |
| Add `llm_model_id`, `llm_region` | `core/config.py` | Modify |
| New FastAPI dependency | `api/middleware/llm_dependencies.py` | New |
| Update `FakeLLMProvider` + new retry tests | `tests/test_llm_provider.py` | Modify |
| Add `tenacity>=0.9.0` | `pyproject.toml` | Modify |

`main.py`, all routes, and all Phase 3 files are **untouched**.

---

## 2. Protocol Signature Update

### `domain/ports.py` — `LLMProvider.generate()`

```python
def generate(
    self,
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = 512,
) -> str: ...
```

The `*` makes `system_prompt` and `max_tokens` keyword-only. All existing
callers using `provider.generate("some prompt")` continue to work — the
new params are purely additive.

---

## 3. Retry Logic

### Library: `tenacity`

`tenacity` is the standard Python retry library. Chosen over a manual
`while` loop because:
- Declarative policy is easier to read and audit.
- Built-in `wait_none()` makes tests instant without patching `time.sleep`.
- The `_wait_strategy` constructor parameter lets us inject `wait_none()`
  in tests, keeping the retry path fully exercised without real sleeps.

### Retry policy

```
stop  = stop_after_attempt(3)          # 1 initial + 2 retries
wait  = wait_exponential(multiplier=0.5, min=0.5, max=4)
retry = retry_if_exception(is_retryable)
```

### `is_retryable(exc)` helper

Checks the exception message (and `.response["Error"]["Code"]` for
`ClientError`) for known retryable strings:
- `"ThrottlingException"` — rate limit
- `"ServiceUnavailableException"` — transient outage
- `"RequestTimeout"` — network timeout

Non-retryable (fail immediately):
- `"AccessDeniedException"` — model not enabled / wrong permissions
- `"ValidationException"` — bad request format

**Why string-matching?** `botocore` exception classes are dynamically
generated — you cannot import `ThrottlingException` directly. The
conventional pattern is to check `exc.response["Error"]["Code"]` for
`ClientError` subclasses, with a `str(exc)` fallback for other types.

### Testable wait via constructor injection

```python
class BedrockLLMProvider:
    def __init__(
        self,
        ...,
        _wait_strategy: Any = wait_exponential(multiplier=0.5, min=0.5, max=4),
    ) -> None:
```

Tests pass `_wait_strategy=wait_none()` — the full retry path runs,
zero sleeps occur.

---

## 4. Extended `converse()` call

```python
request: dict[str, Any] = {
    "modelId": self._model_id,
    "messages": [{"role": "user", "content": [{"text": prompt}]}],
    "inferenceConfig": {"maxTokens": max_tokens},
}
if system_prompt:
    request["system"] = [{"text": system_prompt}]
```

The `system` field is a list of content blocks — Bedrock's format for
Claude's system prompt via `converse()`.

---

## 5. FastAPI Dependency

### `api/middleware/llm_dependencies.py`

```python
def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMProvider:
    return BedrockLLMProvider(
        region=settings.llm_region,
        model_id=settings.llm_model_id,
    )
```

Phase 5 routes use:
```python
llm: Annotated[LLMProvider, Depends(get_llm_provider)]
```

Pattern identical to `get_auth_service()` and `get_retrieval_service()`.

---

## 6. Test Strategy for Retry Logic

Fake boto3 client injected via constructor:

```python
class _FailThenSucceedClient:
    def __init__(self, fail_times: int, success_response: dict) -> None: ...
    def converse(self, **kwargs) -> dict: ...  # raises N times then returns
```

Provider constructed with `client=fake_client, _wait_strategy=wait_none()`
— the full tenacity retry path executes, no sleep.

---

## 7. File Changeset Summary

### New files
```
docs/phases/04-llm-gateway/requirements-piece3.md
docs/phases/04-llm-gateway/design-piece3.md
src/clinical_platform/api/middleware/llm_dependencies.py
```

### Modified files
```
src/clinical_platform/domain/ports.py
src/clinical_platform/infrastructure/llm_providers/bedrock_llm_provider.py
src/clinical_platform/core/config.py
tests/test_llm_provider.py
pyproject.toml
```

### Untouched
```
main.py, all routes, all Phase 3 files,
test_auth.py, test_chunker.py, test_embedding.py, test_retrieval.py
```
