"""Application entry point.

Run with:
    uvicorn clinical_platform.main:app --reload
"""

from fastapi import FastAPI
from clinical_platform.api.routes import observability

from clinical_platform.api.routes import auth as auth_router
from clinical_platform.api.routes import rag as rag_router
from clinical_platform.api.routes import retrieval as retrieval_router

app = FastAPI(
    title="Clinical Intelligence Platform",
    description="Enterprise RAG + Agents platform for pharma/healthcare.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router.router)
app.include_router(retrieval_router.router)
app.include_router(rag_router.router)
app.include_router(observability.router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"], summary="Health check")
def health() -> dict[str, str]:
    return {"status": "ok"}
