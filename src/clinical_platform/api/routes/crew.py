"""
POST /crew/query - runs the 3-agent CrewAI team and returns each
specialist's individual output, so the UI can show the handoff
between Researcher -> Compliance Reviewer -> Writer visually.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from clinical_platform.agents.crews.clinical_crew import build_crew
from clinical_platform.agents.tools.clinical_tools import set_retrieval_service
from clinical_platform.api.middleware.auth_dependencies import CurrentUser
from clinical_platform.core.config import Settings, get_settings
from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
    BedrockEmbeddingProvider,
)
from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
from clinical_platform.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/crew", tags=["crew"])


class CrewQueryRequest(BaseModel):
    question: str


class AgentOutput(BaseModel):
    agent_role: str
    output: str


class CrewQueryResponse(BaseModel):
    final_answer: str
    agent_outputs: list[AgentOutput]


@router.post("/query", response_model=CrewQueryResponse)
def query_crew(
    request: CrewQueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _: CurrentUser,
) -> CrewQueryResponse:
    embedder = BedrockEmbeddingProvider(region=settings.bedrock_region)
    store = JsonVectorStore(store_path=Path(settings.vector_store_path))
    retrieval = RetrievalService(embedder=embedder, store=store)
    set_retrieval_service(retrieval)

    crew = build_crew(question=request.question)
    result = crew.kickoff()

    agent_outputs = [
        AgentOutput(agent_role=task_output.agent, output=str(task_output.raw))
        for task_output in result.tasks_output
    ]

    return CrewQueryResponse(final_answer=str(result.raw), agent_outputs=agent_outputs)