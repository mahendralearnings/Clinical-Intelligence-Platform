"""Retrieval routes: POST /retrieval/ingest and POST /retrieval/search."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from clinical_platform.api.middleware.auth_dependencies import require_permission
from clinical_platform.api.schemas.retrieval import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from clinical_platform.core.config import Settings, get_settings
from clinical_platform.domain.models import Permission
from clinical_platform.domain.ports import DimensionMismatchError, EmbeddingError
from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
    BedrockEmbeddingProvider,
)
from clinical_platform.infrastructure.ingestion.pipeline import IngestPipeline
from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
from clinical_platform.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


# ---------------------------------------------------------------------------
# Dependency: build RetrievalService from Settings
# ---------------------------------------------------------------------------


def get_retrieval_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RetrievalService:
    """Construct a RetrievalService wired with Bedrock + JSON store.

    A new instance is created per-request, which is fine because both adapters
    are either stateless callers (Bedrock) or lightweight file readers
    (JsonVectorStore). Promote to app-lifespan singleton if performance matters.
    """
    embedder = BedrockEmbeddingProvider(
        region=settings.bedrock_region,
        model_id=settings.embedding_model_id,
        dimension=settings.embedding_dimension,
    )
    store = JsonVectorStore(store_path=Path(settings.vector_store_path))
    return RetrievalService(embedder=embedder, store=store)


# ---------------------------------------------------------------------------
# POST /retrieval/ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest documents into the vector store",
)
def ingest(
    body: IngestRequest,
    _current_user: Annotated[object, Depends(require_permission(Permission.READ_DOCUMENTS))],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestResponse:
    """Run the Phase 2 ingestion pipeline, embed every chunk, and upsert into
    the vector store.

    Uses ``body.docs_dir`` if provided, otherwise falls back to
    ``Settings.docs_dir``.  Returns the total number of chunks now in the
    store so the caller can verify the operation.
    """
    docs_path = Path(body.docs_dir) if body.docs_dir else Path(settings.docs_dir)

    if not docs_path.exists():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"docs_dir '{docs_path}' does not exist.",
        )

    pipeline = IngestPipeline(docs_dir=docs_path)
    chunks = pipeline.run()

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No supported documents (.md, .txt) found in '{docs_path}'.",
        )

    try:
        count = retrieval_service.ingest(chunks)
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding provider error: {exc}",
        ) from exc

    return IngestResponse(chunks_ingested=count)


# ---------------------------------------------------------------------------
# POST /retrieval/search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic search over ingested documents",
)
def search(
    body: SearchRequest,
    _current_user: Annotated[object, Depends(require_permission(Permission.READ_DOCUMENTS))],
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> SearchResponse:
    """Embed *query*, compute cosine similarity against stored chunks, and
    return the top-K most relevant passages.
    """
    try:
        results = retrieval_service.search(query=body.query, top_k=body.top_k)
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding provider error: {exc}",
        ) from exc
    except DimensionMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store dimension mismatch: {exc}",
        ) from exc

    return SearchResponse(
        results=[
            SearchResultItem(
                content=chunk.content,
                source=chunk.metadata.source,
                section_title=chunk.metadata.section_title,
                chunk_index=chunk.metadata.chunk_index,
                score=round(score, 6),
            )
            for chunk, score in results
        ]
    )
