"""FastAPI dependency for the LLM provider.

Follows the same pattern as get_auth_service() (auth_dependencies.py) and
get_retrieval_service() (retrieval route).

Phase 5 routes use it as:
    llm: Annotated[LLMProvider, Depends(get_llm_provider)]
"""

from typing import Annotated

from fastapi import Depends

from clinical_platform.infrastructure.llm_providers.anthropic_llm_provider import AnthropicLLMProvider
from clinical_platform.infrastructure.llm_providers.openai_llm_provider import OpenAILLMProvider
from clinical_platform.core.config import Settings, get_settings
from clinical_platform.domain.ports import LLMProvider
from clinical_platform.infrastructure.llm_providers.bedrock_llm_provider import (
    BedrockLLMProvider,
)


# def get_llm_provider(
#     settings: Annotated[Settings, Depends(get_settings)],
# ) -> LLMProvider:
#     """Construct a BedrockLLMProvider from application Settings.

#     A new instance is created per-request. The boto3 client inside
#     BedrockLLMProvider is lazy, so there is no real I/O cost at
#     construction time.

#     Returns:
#         A BedrockLLMProvider configured with llm_region and llm_model_id
#         from Settings (overridable via environment variables / .env).
#     """
#     return BedrockLLMProvider(
#         region=settings.llm_region,
#         model_id=settings.llm_model_id,
#     )



# def get_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
#     if settings.llm_provider_type == "anthropic":
#         return AnthropicLLMProvider(api_key=settings.anthropic_api_key)
#     return BedrockLLMProvider(region=settings.llm_region, model_id=settings.llm_model_id)

def get_llm_provider(settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    if settings.llm_provider_type == "anthropic":
        return AnthropicLLMProvider(api_key=settings.anthropic_api_key)
    if settings.llm_provider_type == "openai":
        return OpenAILLMProvider(api_key=settings.openai_api_key)
    return BedrockLLMProvider(region=settings.llm_region, model_id=settings.llm_model_id)