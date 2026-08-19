"""RAG query route: POST /rag/query — Phase 5."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from clinical_platform.api.middleware.auth_dependencies import require_permission
from clinical_platform.api.middleware.llm_dependencies import get_llm_provider
from clinical_platform.api.routes.retrieval import get_retrieval_service
from clinical_platform.api.schemas.rag import (
    RagQueryRequest,
    RagQueryResponse,
    SourceCitationResponse,
)
from clinical_platform.core.config import Settings, get_settings
from clinical_platform.domain.models import Permission
from clinical_platform.domain.ports import (
    DimensionMismatchError,
    EmbeddingError,
    LLMError,
    LLMProvider,
)
from clinical_platform.services.rag_query_service import RagQueryService
from clinical_platform.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/rag", tags=["rag"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_rag_service(
    retrieval: Annotated[RetrievalService, Depends(get_retrieval_service)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RagQueryService:
    """Wire RagQueryService with retrieval, LLM, and log path from Settings."""
    return RagQueryService(
        retrieval=retrieval,
        llm=llm,
        log_path=Path(settings.query_log_path),
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=RagQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a question using retrieved document chunks",
)
def rag_query(
    body: RagQueryRequest,
    _current_user: Annotated[
        object, Depends(require_permission(Permission.READ_DOCUMENTS))
    ],
    rag_service: Annotated[RagQueryService, Depends(get_rag_service)],
) -> RagQueryResponse:
    """Retrieve relevant chunks and generate a grounded answer via Claude."""
    try:
        result = rag_service.query(question=body.question, top_k=body.top_k)
    except (EmbeddingError, LLMError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except DimensionMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return RagQueryResponse(
        answer=result.answer,
        sources=[
            SourceCitationResponse(
                source=s.source,
                section_title=s.section_title,
                score=s.score,
            )
            for s in result.sources
        ],
    )
