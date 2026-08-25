"""
CrewAI is harder to fully fake than LangGraph (Phase 8) - it manages the
LLM connection internally. So we split testing differently:
  1. Structure tests (free, no AI) - are the right agents/tasks/tools wired?
  2. One optional real run (small cost) - does the crew actually work end-to-end?
"""

import os

import pytest

from clinical_platform.agents.crews.clinical_crew import build_crew


def test_crew_has_three_agents_in_correct_order() -> None:
    crew = build_crew(question="test question")

    roles = [agent.role for agent in crew.agents]
    assert roles == ["Clinical Researcher", "Compliance Reviewer", "Clinical Writer"]


def test_researcher_has_the_search_tool() -> None:
    crew = build_crew(question="test question")

    researcher = crew.agents[0]
    tool_names = [tool.name for tool in researcher.tools]
    assert "Search Clinical Documents" in tool_names


def test_reviewer_task_depends_on_researcher_task() -> None:
    crew = build_crew(question="test question")

    review_task = crew.tasks[1]
    assert crew.tasks[0] in review_task.context


def test_writer_task_depends_on_review_task() -> None:
    crew = build_crew(question="test question")

    writing_task = crew.tasks[2]
    assert crew.tasks[1] in writing_task.context


@pytest.mark.skipif(
    os.getenv("CREW_INTEGRATION") != "1",
    reason="Set CREW_INTEGRATION=1 to run a real, small-cost 3-agent crew",
)
def test_real_crew_answers_a_simple_question() -> None:
    from clinical_platform.agents.tools.clinical_tools import set_retrieval_service
    from clinical_platform.core.config import get_settings
    from clinical_platform.infrastructure.embedding_providers.bedrock_embedding_provider import (
        BedrockEmbeddingProvider,
    )
    from clinical_platform.infrastructure.vector_stores.json_vector_store import JsonVectorStore
    from clinical_platform.services.retrieval_service import RetrievalService
    from pathlib import Path

    settings = get_settings()
    embedder = BedrockEmbeddingProvider(region=settings.bedrock_region)
    store = JsonVectorStore(store_path=Path(settings.vector_store_path))
    set_retrieval_service(RetrievalService(embedder=embedder, store=store))

    crew = build_crew(question="What vitamin deficiency is linked to long-term metformin use?")
    result = crew.kickoff()

    assert "B12" in str(result)